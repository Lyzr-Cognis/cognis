"""
Extraction and decision prompts for Cognis.

Ported verbatim from production lyzr-memory cognis prompts.py.
These exact prompts drive the 79.35% accuracy on LoCoMo.
"""

USER_MEMORY_EXTRACTION_PROMPT = """You are an expert memory extraction assistant. Your task is to extract meaningful, actionable facts from user messages.

**CURRENT DATE:** {current_date}

**INPUT MESSAGE:**
{user_message}

**GUIDELINES FOR EXTRACTION:**

1. **What to Extract (13 Categories):**
   - Identity (names, age, demographics, location)
   - Relationships (family, friends, social connections, pets)
   - Work & Career (job title, company, colleagues, career goals)
   - Learning (education, skills, certifications, languages)
   - Wellness (health, fitness, medical conditions, dietary restrictions)
   - Lifestyle (daily habits, routines, sleep, transportation)
   - Interests (hobbies, passions, entertainment, sports, games)
   - Preferences (likes, dislikes, favorites, style, food)
   - Plans & Goals (future plans, aspirations, upcoming events, trips)
   - Experiences (past events, travel, achievements, milestones)
   - Opinions (views, beliefs, attitudes)
   - Context (session-specific context, current tasks, immediate needs)
   - Miscellaneous (anything that doesn't fit other categories)

2. **Extraction Rules:**
   - Extract ONLY explicit user statements, not inferences
   - Convert relative dates to absolute (e.g., "next week" -> "{current_date} + 7 days")
   - Use the speaker's actual name as subject when mentioned (e.g., "My name is John" -> "John works at..." NOT "User works at...")
   - If no name is mentioned, use "User" as the subject for self-references (I, me, my)
   - Make facts atomic and concise (minimal but complete)
   - Avoid extracting questions, greetings, or acknowledgments

3. **What NOT to Extract:**
   - General knowledge or facts about the world
   - Questions the user asks (no factual content)
   - Small talk, greetings, or pleasantries
   - Transient conversation elements

4. **Output Format:**
Return a JSON object with a "facts" array:
```json
{{"facts": ["Fact 1", "Fact 2", "Fact 3"]}}
```

If no meaningful facts can be extracted, return:
```json
{{"facts": []}}
```

**EXAMPLES:**

Input: "Hi, my name is John and I work at Google as a software engineer"
Output: {{"facts": ["User's name is John", "John works at Google", "John is a software engineer"]}}

Input: "I'm planning to visit Paris next month for my anniversary"
Output: {{"facts": ["User is planning to visit Paris in {next_month}", "User has an anniversary coming up"]}}

Input: "What's the weather like today?"
Output: {{"facts": []}}

Input: "Thanks for your help!"
Output: {{"facts": []}}

Now extract facts from the input message. Return ONLY valid JSON."""


UPDATE_MEMORY_PROMPT = """You are a memory management expert. Your task is to compare NEW FACTS against EXISTING MEMORY and decide what operations to perform.

**EXISTING MEMORY:**
{existing_memory}

**NEW FACTS:**
{new_facts}

**AVAILABLE OPERATIONS:**

1. **ADD** - Use when:
   - The fact is genuinely new information not present in existing memory
   - Adds complementary information (both can coexist)
   - Example: Existing has "likes pizza", new has "favorite color is blue" -> ADD blue fact

2. **UPDATE** - Use when:
   - New fact refines or expands an existing fact
   - New fact clearly supersedes an existing fact: it states the same fact with a newer value or temporal progression
   - Same topic but more detailed information
   - Example: "likes pizza" + "loves pepperoni pizza" -> UPDATE to "loves pepperoni pizza"

   **CRITICAL - UPDATE RULES (REDUCE REDUNDANCY):**
   - UPDATE if 70%+ semantic overlap on same topic/person/event
   - UPDATE if new fact refines or expands existing (prefer consolidation)
   - SKIP if the fact is already substantially captured by existing memory
   - Only ADD for genuinely NEW information not covered by existing memories
   - Prefer UPDATE over ADD when facts are about the same entity

3. **CONTRADICT** - Use when:
   - The new statement and an existing memory make incompatible claims
   - It is NOT clear that the new statement supersedes the old one
   - This includes denials versus assertions or mutually exclusive states asserted about the same period
   - Keep both memories as current so the conflict is preserved
   - Example: "User has a dog" + "User says they do not have a dog" -> CONTRADICT

4. **DELETE** - Use when:
   - The user explicitly asks to forget or retract an existing fact
   - The existing fact is clearly erroneous and should not be retained
   - Do not use DELETE merely because claims are incompatible; use CONTRADICT unless the old fact is clearly superseded
   - **DO NOT DELETE** preferences just because of temporary constraints

5. **NONE** - Use when:
   - Fact already exists in memory (semantically identical)
   - New fact provides no additional information
   - Cosmetic changes only (punctuation, capitalization)

**CRITICAL RULES:**
- Don't delete preferences due to temporary inability (e.g., "can't hike due to injury" doesn't delete "loves hiking")
- When updating, keep the version with MORE information
- Use UPDATE, not CONTRADICT, when the new information clearly supersedes the old information
- Use CONTRADICT, not UPDATE, when incompatible claims cannot be resolved by recency or temporal progression

**OUTPUT FORMAT:**
Return a JSON object with the memory list and operations:
```json
{{
  "memory": [
    {{"id": "existing_id", "text": "existing text", "event": "NONE"}},
    {{"id": "existing_id", "text": "updated text", "event": "UPDATE", "old_text": "previous text"}},
    {{"id": "existing_id", "text": "new fact text", "event": "CONTRADICT"}},
    {{"id": "new", "text": "new fact", "event": "ADD"}},
    {{"id": "existing_id", "text": "deleted text", "event": "DELETE"}}
  ]
}}
```

Analyze the facts and return ONLY valid JSON."""
