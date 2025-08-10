import json

from openai.types.chat import ChatCompletionMessageParam


def prompts_message(data:dict) -> list[ChatCompletionMessageParam]:
    """
    Generate optimized system and user prompts for roommate matching analysis.

    Returns:
        tuple: (system_prompt, user_prompt) - A tuple containing the system prompt and user prompt.
    """
    max_recommendations = 5

    system_prompt = """
The user will provide some exam text. Please parse the "question" and "answer" and output them in JSON format. 

EXAMPLE INPUT: 
Which is the highest mountain in the world? Mount Everest.

EXAMPLE JSON OUTPUT:
{
    "question": "Which is the highest mountain in the world?",
    "answer": "Mount Everest"
}
"""

    user_prompt_prefix = (
        'You are a professional human behavior and personality analyst specializing in roommate compatibility. '
        'Your expertise includes analyzing self-descriptions (both tone and content) to identify personality traits, '
        'living habits, and social preferences. Your primary task is to evaluate compatibility between individuals '
        'as potential roommates based on these insights.'
        'Below are self-descriptions from several students about their ideal dorm living situation. '
        'Please complete the following tasks:\n\n'
        '1. **Comprehensive Analysis:**\n'
        "   - Analyze each student's personality, daily routines, and dorm environment expectations\n"
        '   - Identify key compatibility factors for each individual\n\n'
        f'2. **Roommate Matching (Select 0-{max_recommendations} matches per person):**\n'
        '   Consider these compatibility dimensions:\n'
        '   - **Sleep schedules:** Bedtimes, wake-up times, study/leisure periods\n'
        '   - **Living habits:** Cleanliness standards, organization preferences, noise tolerance\n'
        '   - **Personality compatibility:** Complementary traits that would foster mutual understanding\n'
        '   - **Shared interests:** Common hobbies or activities that could strengthen bonding\n'
        "   - **Reciprocal relationships:** If A is matched to B, B shouldn't also be matched to A\n\n"
        '3. **Output Requirements:**\n'
        '   - Format: Strictly JSON format\n'
        '   - Structure:\n'
        '     * Keys: Student IDs (strings)\n'
        f'     * Values: Arrays of 0-{max_recommendations} recommended roommate IDs\n'
        '   - Validation Rules:\n'
        "     * No self-references (can't recommend oneself)\n"
        '     * No duplicate IDs within recommendations\n'
        '     * All IDs must exist in the input data\n'
        '     * Ensure mutual exclusivity in pairings where possible\n'
        '     * Maintain balanced recommendations across all students\n\n'
        '4. **Additional Guidelines:**\n'
        '   - Prioritize matches with strong alignment in key lifestyle factors\n'
        '   - When preferences conflict, prioritize sleep schedule compatibility\n'
        '   - For students with unique requirements, focus on complementary rather than identical matches'
    )
    
    user_prompt = user_prompt_prefix + '\n\n' + json.dumps(data, ensure_ascii=False, indent=2)

    return [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]


def review_messages(data: dict) -> list[ChatCompletionMessageParam]:
    """
    Generate messages for content review.

    Args:
        data (dict): The data to be reviewed.

    Returns:
        list[ChatCompletionMessageParam]: A list of messages for content review.
    """
    system_prompt = """
      You are a content reviewer. Please review the following data.
      delete what you think is not suitable.
      and output the result in JSON format as the same format as the input.
      If you think the data is suitable, just return the input data.
      """
    user_prompt = json.dumps(data, ensure_ascii=False, indent=2)

    return [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]