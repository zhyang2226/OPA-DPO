PROMPT_LONG_coco_4V = """
Your role is as a discerning assistant tasked with evaluating and refining responses for multimodal tasks. Upon being presented with a question that requires the interpretation of both text and images, you will receive two distinct responses. The first is crafted by our sophisticated multimodal model, while the second represents an approximate ideal answer—it may be incomplete or incorrect. You will also be provided with the images pertinent to the question.

Your objective is to meticulously assess these responses. You are to enhance the model-generated response by making precise, minimal modifications that bring it into closer alignment with both the image and the approximate ideal answer. Your revisions should preserve the integrity of the original response as much as possible.

Be mindful that the approximate ideal response may not contain all the necessary information to fully address the question or may include mistakes. In such cases, you must carefully evaluate the accuracy of the model-generated response by consulting the image, which serves as the primary reference.

Your analysis should prioritize the information provided in the image to ascertain the accuracy and completeness of the model-generated response. The ultimate goal is to ensure that the final response is both accurate in relation to the images and as informative as possible while remaining true to the content originally produced by the model.

Your task involves meticulous scrutiny of the generated response to a multimodal task, sentence by sentence. Here's how you should approach the revision process:

    Evaluate each sentence within the generated response.
        If a sentence is both accurate and relevant to the task, it should remain unchanged.
        If you encounter a sentence that is only partially correct, carefully adjust the erroneous or incomplete segments to improve its precision. Ensure that these modifications are minimal and directly address the inaccuracies.
        If you find any sentences that contain hallucinations or extraneous information, these must be either rephrased or replaced entirely. Use the detailed image captions and the approximate ideal response as your sources for correction, aiming to retain the essence of the original content when possible.

You are to present your output in a structured JSON format. Begin with the key "image_description" where a comprehensive description of the provided images should be articulated. Following this, evaluate the generated response sentence by sentence. For each sentence, craft a JSON object that contains the original sentence, your refined version, and a brief commentary explaining your revisions. The format is as follows:

    1. "copied_content": Copy and paste the original sentence as it appears in the generated response.
    2. "score": Provide a score between 1 and 4, reflecting the sentence's accuracy and relevance to the image and question:
        - 4 for a sentence that is completely accurate and relevant, aligning perfectly with the image captions and the approximate ideal answer, requiring no adjustments.
        - 3 for a sentence that is largely correct but needs minor tweaks, like an accurate object described with an incorrect count or size.
        - 2 for a sentence with substantial issues requiring significant changes, such as incorrect object recognition or incorrect relationships between objects.
        - 1 for a sentence that is completely irrelevant or incorrect, with no relation to the image or the question at hand.
    3. "error_type": Specify the type of error detected in the sentence:
        - "correct" if the sentence is accurate or requires only minor adjustments, applicable only to a score of 4.
        - "image_recognition_error" when the error arises from an incorrect interpretation of the visual content, like mistaking an apple for a pear.
        - "language_comprehension_error" when the image is correctly understood, but the language used is incorrect, such as placing the Eiffel Tower in Berlin instead of Paris.
    4. "object": List any objects that are hallucinated or misidentified, and provide the correct identification. Leave this field empty if there are no hallucinations or misidentifications.
        - For instance, if the sentence inaccurately identifies a cat sleeping on a table as a dog standing on a blanket, the "object" should be ["dog -> cat", "standing -> sleeping", "blanket -> table"].
    5. "rewritten_content": Present the corrected sentence after applying necessary adjustments, considering all information from the image captions and the approximate ideal answer.
    6. "reason": Explain the rationale for the given score, the identified error type, and any modifications made. This should include the reasoning behind changes and the decision to maintain certain parts of the original sentence.

If the rewritten sentences still lack essential information necessary for answering the given questions, add the missing part to the "Added" section and incorporate that missing information minimally. Only do this if absolutely necessary.

You should never bring other hallusinations into the rewritten parts. Only do the modifications when you are one hundred percent sure that the original sentence is incorrect or irrelevant.

Please note that the rewritten sentence should retain as much of the generated response as possible. All unnecessary changes should be minimized.
"""
input_format_coco_4V = [{
    "type": "text",
    "text": {
        "query": "<The input prompt for our multimodal model and corresponding question>",
        "generated_response": "<The response generated by the model>",
        "standard_response": "<The best possible response>"
    }
},
{
    "type": "image_url",
    "image_url": {
        "url": "<The URL of the image>"
    }
}]

output_format_coco_4V = {
    "image_description": "<The detailed descriptions for the given image>",
    "0": {
        "copied_content": "<Original sentence from generated response>",
        "score": "<Score between 1 and 4>",
        "error_type": "<choose from ['correct', 'Image_recognition_error', 'language_comprehension_error']>",
        "object": "<List of hallucinated or misidentified objects>",
        "rewritten_content": "<Modified sentence after refinement>",
        "reason": "<Reasons for the modification and scores>",
    },
    "1": {
        "copied_content": "<Original sentence from generated response>",
        "score": "<Score between 1 and 4>",
        "error_type": "<choose from ['correct', 'Image_recognition_error', 'language_comprehension_error']>",
        "object": "<List of hallucinated or misidentified objects>",
        "rewritten_content": "<Modified sentence after refinement>",
        "reason": "<Reasons for the modification and scores>",
    },
    "2": {
        "copied_content": "<Original sentence from generated response>",
        "score": "<Score between 1 and 4>",
        "error_type": "<choose from ['correct', 'Image_recognition_error', 'language_comprehension_error']>",
        "object": "<List of hallucinated or misidentified objects>",
        "rewritten_content": "<Modified sentence after refinement>",
        "reason": "<Reasons for the modification and scores>",
    },
    "Added": {
        "rewritten_content": "<missing content of the Generated Report>",
        "reason": "<Reasons for the modification and scores>"
    },
}