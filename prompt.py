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
{
  "1804": "同学间能有共同的兴趣爱好与奋斗目标，保持较轻松愉悦的学习与休闲氛围",
  "1805": "大家和睦相处，能成为好友，有共同话题，聊天不会尴尬，一起求学",
  "1806": "舍友之间互相帮助团结友爱，能促进学习一起进步，也能一起玩一起组织活动，作息规律健康生活",
  "1807": "(空)",
  "1808": "大家是互相包容，共同进步，关系亲密的好朋友",
  "1809": "交流学习，适当（可以共同）娱乐",
  "1810": "舍友间相互理解尊重 和谐共处 维持个人空间 有学习的劲头",
}

EXAMPLE JSON OUTPUT:
{
    ...
    "1807":[],
    "1808": [
    1805,
    1809,
    ...
  ],
  ...
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

