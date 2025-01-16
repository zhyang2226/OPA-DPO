import os
import hashlib
import sqlite3
import json
import openai
import time
import logging
import traceback
from threading import Lock

lock = Lock()

logger = logging.getLogger(__name__)

class SqliteStore(object):

    def __init__(self, path):
        self.conn = sqlite3.connect(path)
        cur = self.conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS llm(cache_key, response)")
        cur.execute("CREATE TABLE IF NOT EXISTS key_store(key, data)")
        self.conn.commit()

    def save_llm_response(self, cache_key, response):
        data = [
            (cache_key, response),
        ]
        cur = self.conn.cursor()
        cur.executemany("INSERT INTO llm(cache_key, response) VALUES(?, ?)", data)
        self.conn.commit()

    def get_llm_response_by_cache_key(self, cache_key):
        cur = self.conn.cursor()
        ret = cur.execute("SELECT response FROM llm WHERE cache_key= ?", (cache_key,)).fetchone()
        if ret is None:
            return None
        else:
            return ret[0]

    def save_object_by_key(self, key, data):
        jdata = json.dumps(data)
        cur = self.conn.cursor()
        ret = cur.execute("SELECT data FROM key_store WHERE key = ?", (key,)).fetchone()
        if ret is None:
            cur.execute("INSERT INTO key_store(key, data) VALUES(?, ?)", (key, jdata))
        else:
            cur.execute("UPDATE key_store SET data = ? WHERE key = ?", (jdata, key))
        self.conn.commit()

    def get_object_by_key(self, key):
        cur = self.conn.cursor()
        ret = cur.execute("SELECT data FROM key_store WHERE key = ?", (key,)).fetchone()
        if ret is None:
            return None
        else:
            jdata = ret[0]
            return json.loads(jdata)

def get_stable_represent_str(obj):
    if isinstance(obj, int):
        return str(obj)
    elif isinstance(obj, float):
        return str("{:.6f}".format(obj))
    elif isinstance(obj, str):
        return obj
    elif isinstance(obj, list):
        return "#%".join([get_stable_represent_str(entry) for entry in obj])
    elif isinstance(obj, dict):
        desc_list = []
        for k in sorted(obj.keys()):
            v = obj[k]
            desc_list.append("k={}+v={}".format(
                get_stable_represent_str(k),
                get_stable_represent_str(v),
            ))
        return "#%".join(desc_list)
    else:
        raise NotImplementedError

def get_cache_key(obj):
    stable_represent_str = get_stable_represent_str(obj)
    return hashlib.md5(stable_represent_str.encode('utf-8')).hexdigest()

class APIService(object):

    def __init__(self, client, history_store_path, max_retries=3, use_cache=True):

        self.store = SqliteStore(history_store_path)
        self.max_retries = max_retries
        self.openai_embedding_provider = None
        self.client = client
        self.use_cache = use_cache

    def call_llm_with_messages(self, messages, high_or_low_temp='low', model="gpt-3.5-turbo-1106", max_tokens=3000):
        retries = 0
        while True:
            try:
                if high_or_low_temp == 'low':
                    temp = 0.0
                elif high_or_low_temp == "high":
                    temp = 1.0
                else:
                    raise NotImplementedError
                call_args = {
                    "model": model,
                    "temperature": temp,
                    "max_tokens": max_tokens,
                    "top_p": 0.95,
                    "frequency_penalty": 0,
                    "presence_penalty": 0,
                    "stream": False,
                }
                cache_key = get_cache_key({
                    "type": "llm",
                    "messages": messages,
                    "call_args": call_args,
                })
                cached_response = self.store.get_llm_response_by_cache_key(cache_key)
                if self.use_cache is False:
                    cached_response = None
                if cached_response is None:
                    with lock:
                        response = self.client.chat.completions.create(
                            messages=messages,
                            **call_args,
                        )
                    response_text = response.choices[0].message.content
                    if self.use_cache:
                        self.store.save_llm_response(cache_key, response_text)
                    return response_text
                else:
                    return cached_response
            except Exception as e:
                error_message = str(e)
                logger.error(traceback.format_exc())
                global_rank = int(os.environ.get("RANK", 0))
                
                print(f"RANK{global_rank} Retry {retries}: Rate limit exceeded. Retrying after a short delay...")
                time.sleep(6)
                retries += 1
                if retries >= self.max_retries:
                    raise e

    def call_llm_with_prompt(self, prompt, high_or_low_temp='low', model="gpt-3.5-turbo-1106", max_tokens=3000):
        messages = [
            {"role": "system", "content": "You are an helpful AI assistant."},
            {"role": "user", "content": prompt},
        ]
        return self.call_llm_with_messages(messages, high_or_low_temp, model, max_tokens)

    # def call_openai_embeddings_for_many(self, messages):
    #     if self.openai_embedding_provider is None:
    #         from f import OpenAIEmbeddingProvider
    #         self.openai_embedding_provider = OpenAIEmbeddingProvider({})
    #     cache_key = get_cache_key({
    #         "type": "openai_embedding",
    #         "messages": messages,
    #     })
    #     cached_response = self.store.get_object_by_key(cache_key)
    #     if self.use_cache is False:
    #         cached_response = None
    #     if cached_response is None:
    #         response = self.openai_embedding_provider.get_embeddings_for_many(messages)
    #         if self.use_cache:
    #             self.store.save_object_by_key(cache_key, response)
    #         return response
    #     else:
    #         return cached_response


def get_api_service(type='azure', key="", max_retries=10, use_cache=True, azure_endpoint="https://gcrendpoint.azurewebsites.net"):
    from openai import AzureOpenAI
    from openai import OpenAI

    if type == "azure":
        client = AzureOpenAI(
            # api_version="2023-07-01-preview",
            api_version='2024-02-01',
            azure_endpoint=azure_endpoint,
            api_key=key,
        )
    elif type == "openai":
        client = OpenAI(api_key=key)
    else:
        raise NotImplementedError

    api_service = APIService(client, "history.sqlite", max_retries=max_retries, use_cache=use_cache)
    return api_service
