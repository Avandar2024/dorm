def generate_roommate_matching_prompts():
    """
    Generate optimized system and user prompts for roommate matching analysis.
    
    Returns:
        tuple: (system_prompt, user_prompt) - A tuple containing the system prompt and user prompt.
    """
    min_recommendations = 3
    max_recommendations = 5
    
    system_prompt = (
        "You are a professional human behavior and personality analyst specializing in roommate compatibility. "
        "Your expertise includes analyzing self-descriptions (both tone and content) to identify personality traits, "
        "living habits, and social preferences. Your primary task is to evaluate compatibility between individuals "
        "as potential roommates based on these insights."
    )
    
    user_prompt = (
        "Below are self-descriptions from several students about their ideal dorm living situation. "
        "Please complete the following tasks:\n\n"
        
        "1. **Comprehensive Analysis:**\n"
        "   - Analyze each student's personality, daily routines, and dorm environment expectations\n"
        "   - Identify key compatibility factors for each individual\n\n"
        
        f"2. **Roommate Matching (Select {min_recommendations}-{max_recommendations} matches per person):**\n"
        "   Consider these compatibility dimensions:\n"
        "   - **Sleep schedules:** Bedtimes, wake-up times, study/leisure periods\n"
        "   - **Living habits:** Cleanliness standards, organization preferences, noise tolerance\n"
        "   - **Personality compatibility:** Complementary traits that would foster mutual understanding\n"
        "   - **Shared interests:** Common hobbies or activities that could strengthen bonding\n"
        "   - **Reciprocal relationships:** If A is matched to B, B shouldn't also be matched to A\n\n"
        
        "3. **Output Requirements:**\n"
        "   - Format: Strictly JSON format\n"
        "   - Structure:\n"
        "     * Keys: Student IDs (strings)\n"
        f"     * Values: Arrays of {min_recommendations}-{max_recommendations} recommended roommate IDs\n"
        "   - Validation Rules:\n"
        "     * No self-references (can't recommend oneself)\n"
        "     * No duplicate IDs within recommendations\n"
        "     * All IDs must exist in the input data\n"
        "     * Ensure mutual exclusivity in pairings where possible\n"
        "     * Maintain balanced recommendations across all students\n\n"
        
        "4. **Additional Guidelines:**\n"
        "   - Prioritize matches with strong alignment in key lifestyle factors\n"
        "   - When preferences conflict, prioritize sleep schedule compatibility\n"
        "   - For students with unique requirements, focus on complementary rather than identical matches"
    )
    
    return system_prompt, user_prompt