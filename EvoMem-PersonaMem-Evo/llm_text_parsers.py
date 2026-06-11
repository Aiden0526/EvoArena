"""
Plain-text prompt templates, section-marker parsers, and validation logic
for the robust A-MEM system. Replaces JSON-schema LLM calls with plain-text
prompts that work with any LLM backend (Ollama, SGLang, OpenAI, etc.).
"""

import json
import re
import logging
from typing import Dict, List, Any, Optional, Callable

logger = logging.getLogger("amem_robust")

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` fences from LLM output."""
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?\s*```$', '', text, flags=re.MULTILINE)
    return text.strip()


def parse_with_json_fallback(response: str, plain_text_parser: Callable, *parser_args) -> Any:
    """Try JSON parsing first; fall back to section-marker parsing.

    Many models emit valid JSON even without strict mode, so we try that first
    for best-of-both-worlds compatibility.
    """
    try:
        cleaned = strip_markdown_fences(response)
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    return plain_text_parser(response, *parser_args)


# ---------------------------------------------------------------------------
# List parsing helpers
# ---------------------------------------------------------------------------

def _parse_list_items(text: str) -> List[str]:
    """Parse a section of text into a list of items.

    Handles:
      - Bullet points (-, *, numbered)
      - Comma-separated values
      - One item per line
    """
    if not text or not text.strip():
        return []

    lines = text.strip().splitlines()
    items: List[str] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Strip bullet markers
        line = re.sub(r'^[\-\*\u2022]\s*', '', line)
        line = re.sub(r'^\d+[\.\)]\s*', '', line)
        # Strip surrounding quotes
        line = line.strip().strip('"').strip("'").strip()
        if not line:
            continue
        # If the line contains commas, split on them
        if ',' in line:
            for part in line.split(','):
                part = part.strip().strip('"').strip("'").strip()
                if part:
                    items.append(part)
        else:
            items.append(line)

    return items


def _extract_section(text: str, marker: str, next_markers: Optional[List[str]] = None) -> str:
    """Extract the text between *marker*: and the next known marker (or end).

    Args:
        text: Full LLM response
        marker: Section header to find (e.g. "KEYWORDS")
        next_markers: List of possible next section headers

    Returns:
        The text content of that section (may be empty string).
    """
    # Build a regex that finds the marker (case-insensitive) followed by a colon
    pattern = re.compile(
        rf'^\s*{re.escape(marker)}\s*:\s*(.*)$',
        re.IGNORECASE | re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        return ""

    start = match.end()
    # The first line of content may be on the same line as the marker
    first_line = match.group(1).strip()

    # Find where the next section starts
    end = len(text)
    if next_markers:
        for nm in next_markers:
            nm_pattern = re.compile(
                rf'^\s*{re.escape(nm)}\s*:', re.IGNORECASE | re.MULTILINE
            )
            nm_match = nm_pattern.search(text, start)
            if nm_match and nm_match.start() < end:
                end = nm_match.start()

    rest = text[start:end].strip()
    if first_line and rest:
        return first_line + "\n" + rest
    return first_line or rest


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

ANALYZE_CONTENT_PROMPT = """Analyze the following content and provide:
1. KEYWORDS: The most important keywords (nouns, verbs, key concepts). Order from most to least important. At least three keywords. Do not include speaker names or time references.
2. CONTEXT: One sentence summarizing the main topic, key points, and purpose.
3. TAGS: Broad categories/themes for classification (domain, format, type). At least three tags.

Respond using EXACTLY this format (one section per header):

KEYWORDS: keyword1, keyword2, keyword3, ...
CONTEXT: A single sentence summarizing the content.
TAGS: tag1, tag2, tag3, ...

Content for analysis:
{content}"""


ANALYZE_CONTENT_PREF_PROMPT = """You are a memory analyst extracting personal preferences from conversation turns.

CRITICAL INSIGHT: Preference changes are NEVER announced explicitly ("I changed my preference from X to Y"). They are embedded as the natural context or framing of a practical request. You must infer the change from HOW the user describes their current situation, not from what they literally say about changing.

── CHANGE TYPE DETECTION GUIDE ──────────────────────────────────────────────

▸ same_object_flip  — attitude polarity reversed toward the SAME object/topic
  The signal is surprise, re-engagement, or disengagement framed as recent/current.
  Like→Dislike patterns:
    "I used to be able to appreciate X more, and now even hearing about it bothers me"
    "I keep seeing [X] and honestly I'd rather avoid it"
    "I need something that doesn't turn into [drawn-out X]"  ← implicit dislike
    "I do better without [X]" | "X feels exhausting now"
  Dislike→Like patterns (CRITICAL — these are the hardest to detect):
    SURPRISE MARKERS signal that this interest is unexpected — i.e., the user previously did not care about or disliked this topic:
    "I've been weirdly interested lately in X" ← 'weirdly' = unexpected → this is a FLIP, NOT a new preference
    "I went down a rabbit hole about X and now I'm curious how it works"  ← 'now' after rabbit hole = flip
    "I realize this is niche, but I've been into X lately"  ← "I realize this is niche" = self-aware of unexpected interest
    "I find X calming in a way I didn't expect"  ← "didn't expect" = surprise = implicit flip
    "Oddly enough, I've started enjoying X" | "I never thought I'd say this but X is growing on me"
  DECISION RULE for Dislike→Like: if the user uses surprise/self-deprecating language about their OWN interest
  ("weirdly", "oddly", "unexpectedly", "I realize this is niche", "I didn't expect", "somehow I've been into"),
  classify as same_object_flip — NOT new_preference. The surprise implies a prior neutral/negative stance.
  Key: the user references the SAME topic but their current stance clearly differs from a prior stance implied in the content.

▸ object_replacement  — switched to a DIFFERENT alternative in the SAME category
  The signal is present-tense mention of a NEW item/choice that implicitly replaced an older one.
  The old preference is rarely mentioned — only the new current state is described.
    "I'm on an SNRI right now and it helps some" ← implies switched from something else
    "I've been reading more long-form pieces on climate economics lately"
    "I've been reading a lot of long-form work on transit and zoning"
    "I'd be a lot more excited by a place with a strong live music culture than by galleries"  ← explicit contrast
    "I realized I'd prefer X over Y" | "I've switched to X" | "instead of my usual X"
  Key: current-tense engagement with X where X clearly belongs to a category the user has preferences about.

▸ conditional_preference  — the preference now APPLIES ONLY under a specific condition
  The condition is embedded as a qualifier WITHIN the preference statement itself.
    "ask my doctor to explain things more directly, but preferably when it's just me in the room"
    "I do best with very clear communication... but I especially notice it on bigger occasions"
    "I like having a full game on when I can, but I'm trying to be careful about extra spending"
    "when I'm by myself and already feeling raw or foggy, I do much better with something intuitive"
    "I prefer X but only when Y" | "X works for me when Z" | "I enjoy X, especially when W"
  Key: preference is stated WITH a qualifying situation/condition ("but when", "especially when", "but only when", "preferably when", "when I'm [state]").

▸ attribute_swap  — same domain/topic, different specific attribute, variant, or style
  The user expresses preference for a SPECIFIC ATTRIBUTE within a familiar domain, often contrasting:
    "I really do better with quieter trips that are more scenic than busy"  ← scenic/quiet vs busy
    "I need something that feels warm, lived-in... rather than stark or warehouse-like"  ← attribute contrast
    "I think I need things that are more therapist-led or personal story-based right now"
    "I do better when someone texts me same-day... than when I have a packed weekend"  ← spontaneous vs scheduled
    "[new attr] rather than [old attr]" | "more [X] than [Y]" | "not [old] but [new]"
  Key: same preference domain, but the specific variant/attribute/style has shifted.

▸ temporal_validity  — preference now APPLIES ONLY during a specific time period
  The time period appears as the FRAMING CONTEXT of an otherwise normal request. The preference itself has not changed — only WHEN it applies has shifted.
  Patterns: the time word is baked directly into the request topic or opening phrase.
    "Do you know of any good ways to find music for the evening based on whatever mood I'm in?"
      → 'for the evening' frames the whole request — temporal_validity
    "At night I don't really want to force myself into one genre"
      → 'at night' scopes the music preference — temporal_validity
    "Can you give me good strategy board game recommendations for winter?"
      → 'for winter' is the main context — temporal_validity
    "Do you have any suggestions for weekend playlists or artists..."
      → 'weekend' is the framing — temporal_validity
    "I'm trying to get ahead of a few bigger household purchases this spring"
      → 'this spring' scopes the budgeting/shopping preference — temporal_validity
    "I'm putting together a holiday playlist" | "I have a quiet weekend coming up"
      → time occasion frames the preference request — temporal_validity
    "Once it gets cold out, I always end up circling back to familiar favorites"
      → 'cold out / winter' scopes the comfort-rewatch preference — temporal_validity
  DETECTION RULE: If a preference request is naturally framed inside a time word (season, time-of-day, day-of-week, holiday, occasion) and that time word is the primary context driving the request, classify as temporal_validity.
  The time word does NOT need to appear as an explicit restriction ("only in winter") — it just needs to be the frame ("for winter", "in the morning", "on weekends", "during the holiday").

▸ ask_to_forget  — explicit COMMAND to the assistant to ERASE a preference from memory
  STRUCTURAL SIGNATURE: Almost always follows an assistant turn that cited a preference.
  The user issues a memory-management instruction, NOT a new preference statement.
  Canonical form: "Please forget that I [like/have/enjoy/prefer/use] X"
  Also: "Don't remember that I like X" | "Remove from your memory that I prefer Y"
         "Stop treating me as someone who likes X" | "I don't want you to keep that in mind"
  ── NOT ask_to_forget ──
    "I used to like X but now I prefer Y" → object_replacement
    "I don't enjoy X as much anymore"     → same_object_flip
    "I've switched from X to Y"           → object_replacement
    "Coffee doesn't agree with me like it used to" → same_object_flip
  ask_to_forget is ALWAYS a direct imperative command about erasing a memory entry.

── OUTPUT INSTRUCTIONS ──────────────────────────────────────────────────────

Analyze the content and provide:
1. KEYWORDS: Most important keywords. At least three. No speaker names or time references.
2. CONTEXT: One sentence summarizing the content.
   - If IS_PREFERENCE is YES: MUST state the preference explicitly ("User prefers X", "User dislikes Y").
   - If CHANGE_TYPE is not none/new_preference: MUST encode the transition direction:
     same_object_flip:       "User now [likes/dislikes] X (previously [opposite])."
     object_replacement:     "User now prefers Y [instead of/over] X in [domain]."
     conditional_preference: "User's preference for X is now conditional: applies [condition]."
     attribute_swap:         "User now prefers [new attr] over [old attr] in [domain]."
     temporal_validity:      "User's preference for X now applies [time period]."
     ask_to_forget:          "User asked to forget/remove their preference for X."
3. TAGS: Broad categories/themes. At least three. Include "pref:[domain]" if a preference is found.
4. IS_PREFERENCE: YES or NO — does this content express, update, or revoke a personal preference?
5. PREF_DOMAIN: 2-5 word topic domain if IS_PREFERENCE=YES, else NONE.
6. PREF_HOLDER: self | others | none
7. CHANGE_TYPE: same_object_flip | object_replacement | conditional_preference | attribute_swap | temporal_validity | ask_to_forget | new_preference | none
   RULE: Only use ask_to_forget when the user commands the assistant to erase a memory entry. Never use it for natural preference change statements.

Respond using EXACTLY this format:

KEYWORDS: keyword1, keyword2, keyword3, ...
CONTEXT: A single sentence summarizing the content.
TAGS: tag1, tag2, tag3, ...
IS_PREFERENCE: YES or NO
PREF_DOMAIN: domain or NONE
PREF_HOLDER: self or others or none
CHANGE_TYPE: same_object_flip | object_replacement | conditional_preference | attribute_swap | temporal_validity | ask_to_forget | new_preference | none

Content for analysis:
{content}"""


EVOLUTION_DECISION_PROMPT = """You are an AI memory evolution agent. Analyze the new memory note and its nearest neighbors to decide if evolution is needed.

New memory:
- Context: {context}
- Content: {content}
- Keywords: {keywords}

Nearest neighbor memories:
{nearest_neighbors_memories}

Based on the relationships between the new memory and its neighbors, decide:
- NO_EVOLUTION: The memory stands alone, no changes needed.
- STRENGTHEN: The new memory should be linked to some neighbors and its tags updated.
- UPDATE_NEIGHBOR: The neighbors' context/tags should be updated based on new understanding.
- STRENGTHEN_AND_UPDATE: Both strengthen and update neighbors.

Respond using EXACTLY this format:
DECISION: <one of NO_EVOLUTION, STRENGTHEN, UPDATE_NEIGHBOR, STRENGTHEN_AND_UPDATE>
REASON: <brief explanation>"""


EVOLUTION_DECISION_PREF_PROMPT = """You are an AI memory evolution agent. Analyze the new memory note and its nearest neighbors to decide if evolution is needed.

New memory:
- Context: {context}
- Content: {content}
- Keywords: {keywords}

Nearest neighbor memories:
{nearest_neighbors_memories}

PREFERENCE EVOLUTION RULES (apply first if the new memory expresses a personal preference):
- If a neighbor covers the SAME preference domain with the SAME value → STRENGTHEN (reinforce the existing preference node).
- If a neighbor covers the SAME preference domain with a DIFFERENT value → UPDATE_NEIGHBOR (the neighbor's context MUST reflect the new preference value, replacing the old one).
- If the new memory asks to forget or revoke a preference held by a neighbor → UPDATE_NEIGHBOR (mark the preference as revoked but preserve the topic and original value for reference).

For non-preference content, use general rules:
- NO_EVOLUTION: The memory stands alone, no changes needed.
- STRENGTHEN: The new memory should be linked to some neighbors and its tags updated.
- UPDATE_NEIGHBOR: The neighbors' context/tags should be updated based on new understanding.
- STRENGTHEN_AND_UPDATE: Both strengthen and update neighbors.

Respond using EXACTLY this format:
DECISION: <one of NO_EVOLUTION, STRENGTHEN, UPDATE_NEIGHBOR, STRENGTHEN_AND_UPDATE>
REASON: <brief explanation>"""


STRENGTHEN_DETAILS_PROMPT = """Given the new memory and its neighbors, provide updated connections and tags.

New memory:
- Content: {content}
- Keywords: {keywords}

Neighbor memories:
{nearest_neighbors_memories}

Which neighbor indices should the new memory connect to? What tags best describe this memory?

Respond using EXACTLY this format:
CONNECTIONS: 0, 2, 3
TAGS: tag1, tag2, tag3, ..."""


STRENGTHEN_DETAILS_PREF_PROMPT = """Given the new memory and its neighbors, provide updated connections and tags.

New memory:
- Content: {content}
- Keywords: {keywords}

Neighbor memories:
{nearest_neighbors_memories}

Which neighbor indices should the new memory connect to? What tags best describe this memory?
When connecting preference-related memories, include a tag for the preference domain (e.g., "pref:dessert_flavors", "pref:outdoor_activities").

Respond using EXACTLY this format:
CONNECTIONS: 0, 2, 3
TAGS: tag1, tag2, tag3, ..."""


UPDATE_NEIGHBORS_PROMPT = """Given the new memory and its neighbor memories, update each neighbor's context and tags based on a holistic understanding of all these memories together.

New memory:
- Content: {content}
- Context: {context}

Neighbor memories:
{nearest_neighbors_memories}

For each neighbor (indexed 0 to {max_neighbor_idx}), provide updated context and tags. If no change is needed, repeat the original values.

Respond using EXACTLY this format (one block per neighbor):

NEIGHBOR 0:
CONTEXT: updated context sentence
TAGS: tag1, tag2, tag3

NEIGHBOR 1:
CONTEXT: updated context sentence
TAGS: tag1, tag2, tag3

(continue for all {neighbor_count} neighbors)"""


UPDATE_NEIGHBORS_PREF_PROMPT = """Given the new memory and its neighbor memories, update each neighbor's context and tags based on a holistic understanding of all these memories together.

New memory:
- Content: {content}
- Context: {context}

Neighbor memories:
{nearest_neighbors_memories}

PREFERENCE UPDATE RULES:
When updating a neighbor that holds a preference:
- If the preference VALUE changed: state the NEW value clearly and explicitly.
  Example: "User now prefers vanilla desserts" — NOT "User has discussed various flavors."
- If the preference was REVOKED (user asked to forget): keep the topic and write "User previously preferred X; preference revoked by user request."
  Do NOT delete the preference information entirely.
- Always preserve the preference domain clearly in the context and tags.

For each neighbor (indexed 0 to {max_neighbor_idx}), provide updated context and tags. If no change is needed, repeat the original values.

Respond using EXACTLY this format (one block per neighbor):

NEIGHBOR 0:
CONTEXT: updated context sentence
TAGS: tag1, tag2, tag3

NEIGHBOR 1:
CONTEXT: updated context sentence
TAGS: tag1, tag2, tag3

(continue for all {neighbor_count} neighbors)"""


FOCUSED_KEYWORDS_PROMPT = """List exactly 5 keywords that capture the main concepts of the following text. Output only the keywords, comma-separated, nothing else.

Text: {content}"""


# ---------------------------------------------------------------------------
# Parsers for each call site
# ---------------------------------------------------------------------------

def parse_analyze_content(response: str, content: str = "") -> Dict[str, Any]:
    """Parse the analyze_content LLM response.

    Returns:
        {"keywords": [...], "context": "...", "tags": [...],
         "is_preference": bool, "pref_domain": str, "pref_holder": str}
    """
    _VALID_CHANGE_TYPES = {
        "same_object_flip", "object_replacement", "conditional_preference",
        "attribute_swap", "temporal_validity", "ask_to_forget",
        "new_preference", "none",
    }

    def _section_parse(resp: str, content_text: str = "") -> Dict[str, Any]:
        _all = ["KEYWORDS", "CONTEXT", "TAGS", "IS_PREFERENCE", "PREF_DOMAIN", "PREF_HOLDER", "CHANGE_TYPE"]
        keywords_text = _extract_section(resp, "KEYWORDS", [s for s in _all if s != "KEYWORDS"])
        context_text = _extract_section(resp, "CONTEXT", [s for s in _all if s != "CONTEXT"])
        tags_text = _extract_section(resp, "TAGS", [s for s in _all if s != "TAGS"])
        is_pref_text = _extract_section(resp, "IS_PREFERENCE", [s for s in _all if s != "IS_PREFERENCE"])
        pref_domain_text = _extract_section(resp, "PREF_DOMAIN", [s for s in _all if s != "PREF_DOMAIN"])
        pref_holder_text = _extract_section(resp, "PREF_HOLDER", [s for s in _all if s != "PREF_HOLDER"])
        change_type_text = _extract_section(resp, "CHANGE_TYPE", [s for s in _all if s != "CHANGE_TYPE"])

        keywords = _parse_list_items(keywords_text)
        context = context_text.strip() if context_text.strip() else ""
        tags = _parse_list_items(tags_text)
        is_preference = is_pref_text.strip().upper() == "YES"
        pref_domain_raw = pref_domain_text.strip()
        pref_domain = pref_domain_raw if pref_domain_raw.upper() not in ("", "NONE") else ""
        pref_holder_raw = pref_holder_text.strip().lower().split()[0] if pref_holder_text.strip() else "none"
        pref_holder = pref_holder_raw if pref_holder_raw in ("self", "others") else "none"
        change_type_raw = change_type_text.strip().lower().split()[0] if change_type_text.strip() else "none"
        # Strip surrounding punctuation/pipes that LLMs sometimes add
        change_type_raw = change_type_raw.strip("|").strip()
        change_type = change_type_raw if change_type_raw in _VALID_CHANGE_TYPES else "none"

        return {
            "keywords": keywords,
            "context": context,
            "tags": tags,
            "is_preference": is_preference,
            "pref_domain": pref_domain,
            "pref_holder": pref_holder,
            "change_type": change_type,
        }

    result = parse_with_json_fallback(response, _section_parse, content)

    # Validate / repair core fields
    result = validate_analysis_result(result, content)
    # Ensure preference fields have defaults if missing (e.g. from JSON fallback)
    result.setdefault("is_preference", False)
    result.setdefault("pref_domain", "")
    result.setdefault("pref_holder", "none")
    result.setdefault("change_type", "none")
    return result


def parse_evolution_decision(response: str) -> Dict[str, str]:
    """Parse the evolution decision response.

    Returns:
        {"decision": "NO_EVOLUTION|STRENGTHEN|UPDATE_NEIGHBOR|STRENGTHEN_AND_UPDATE",
         "reason": "..."}
    """
    def _section_parse(resp: str) -> Dict[str, str]:
        decision_text = _extract_section(resp, "DECISION", ["REASON"])
        reason_text = _extract_section(resp, "REASON", ["DECISION"])

        decision = decision_text.strip().upper().replace(" ", "_")
        # Normalize common variants
        valid_decisions = {
            "NO_EVOLUTION", "STRENGTHEN", "UPDATE_NEIGHBOR",
            "STRENGTHEN_AND_UPDATE"
        }
        if decision not in valid_decisions:
            # Try to infer from keywords
            resp_upper = resp.upper()
            if "STRENGTHEN" in resp_upper and "UPDATE" in resp_upper:
                decision = "STRENGTHEN_AND_UPDATE"
            elif "STRENGTHEN" in resp_upper:
                decision = "STRENGTHEN"
            elif "UPDATE" in resp_upper:
                decision = "UPDATE_NEIGHBOR"
            else:
                decision = "NO_EVOLUTION"

        return {"decision": decision, "reason": reason_text.strip()}

    result = parse_with_json_fallback(response, _section_parse)

    # Map JSON keys if we got JSON
    if "should_evolve" in result:
        should_evolve = result.get("should_evolve", False)
        actions = result.get("actions", [])
        if not should_evolve:
            decision = "NO_EVOLUTION"
        elif "strengthen" in actions and "update_neighbor" in actions:
            decision = "STRENGTHEN_AND_UPDATE"
        elif "strengthen" in actions:
            decision = "STRENGTHEN"
        elif "update_neighbor" in actions:
            decision = "UPDATE_NEIGHBOR"
        else:
            decision = "NO_EVOLUTION"
        result = {"decision": decision, "reason": ""}

    if "decision" not in result:
        result = {"decision": "NO_EVOLUTION", "reason": ""}

    return result


def parse_strengthen_details(response: str) -> Dict[str, Any]:
    """Parse the strengthen details response.

    Returns:
        {"connections": [int, ...], "tags": [str, ...]}
    """
    def _section_parse(resp: str) -> Dict[str, Any]:
        conn_text = _extract_section(resp, "CONNECTIONS", ["TAGS"])
        tags_text = _extract_section(resp, "TAGS", ["CONNECTIONS"])

        # Parse connections as integers
        connections = []
        for item in _parse_list_items(conn_text):
            try:
                connections.append(int(item.strip()))
            except (ValueError, TypeError):
                pass

        tags = _parse_list_items(tags_text)
        return {"connections": connections, "tags": tags}

    result = parse_with_json_fallback(response, _section_parse)

    # Map from JSON keys if needed
    if "suggested_connections" in result and "connections" not in result:
        result["connections"] = [int(x) for x in result.get("suggested_connections", []) if isinstance(x, (int, float))]
    if "tags_to_update" in result and "tags" not in result:
        result["tags"] = result.get("tags_to_update", [])

    result.setdefault("connections", [])
    result.setdefault("tags", [])
    return result


def parse_update_neighbors(response: str, num_neighbors: int) -> List[Dict[str, Any]]:
    """Parse the update neighbors response.

    Returns:
        [{"context": "...", "tags": [...]}, ...] — one per neighbor
    """
    def _section_parse(resp: str, n_neighbors: int) -> List[Dict[str, Any]]:
        neighbors = []
        for i in range(n_neighbors):
            # Try to find NEIGHBOR i: block
            # Look for "NEIGHBOR i:" or "NEIGHBOR i\n"
            pattern = re.compile(
                rf'NEIGHBOR\s+{i}\s*:', re.IGNORECASE
            )
            match = pattern.search(resp)
            if not match:
                neighbors.append({"context": "", "tags": []})
                continue

            # Find the end of this neighbor block (next NEIGHBOR or end)
            next_pattern = re.compile(
                rf'NEIGHBOR\s+{i + 1}\s*:', re.IGNORECASE
            )
            next_match = next_pattern.search(resp, match.end())
            block_end = next_match.start() if next_match else len(resp)
            block = resp[match.end():block_end]

            ctx = _extract_section(block, "CONTEXT", ["TAGS"])
            tags_text = _extract_section(block, "TAGS", ["CONTEXT"])
            tags = _parse_list_items(tags_text)

            neighbors.append({"context": ctx.strip(), "tags": tags})

        return neighbors

    # Try JSON first
    try:
        cleaned = strip_markdown_fences(response)
        data = json.loads(cleaned)
        if isinstance(data, dict):
            contexts = data.get("new_context_neighborhood", [])
            tags_list = data.get("new_tags_neighborhood", [])
            neighbors = []
            for i in range(num_neighbors):
                ctx = contexts[i] if i < len(contexts) else ""
                tags = tags_list[i] if i < len(tags_list) else []
                neighbors.append({"context": ctx, "tags": tags})
            return neighbors
    except (json.JSONDecodeError, ValueError):
        pass

    return _section_parse(response, num_neighbors)


def parse_plain_text_answer(response: str) -> str:
    """Parse a plain-text answer response (for QA evaluation).

    If the model returned JSON with an "answer" field, extract it.
    Otherwise return the raw text.
    """
    try:
        cleaned = strip_markdown_fences(response)
        data = json.loads(cleaned)
        if isinstance(data, dict) and "answer" in data:
            return str(data["answer"])
    except (json.JSONDecodeError, ValueError):
        pass
    return response.strip()


def parse_relevant_parts(response: str) -> str:
    """Parse retrieve_memory_llm response.

    If JSON with "relevant_parts", extract it. Otherwise return raw text.
    """
    try:
        cleaned = strip_markdown_fences(response)
        data = json.loads(cleaned)
        if isinstance(data, dict) and "relevant_parts" in data:
            return str(data["relevant_parts"])
    except (json.JSONDecodeError, ValueError):
        pass
    return response.strip()


def parse_keywords_response(response: str) -> str:
    """Parse generate_query_llm response.

    If JSON with "keywords", extract it. Otherwise return raw text.
    """
    try:
        cleaned = strip_markdown_fences(response)
        data = json.loads(cleaned)
        if isinstance(data, dict) and "keywords" in data:
            return str(data["keywords"])
    except (json.JSONDecodeError, ValueError):
        pass
    return response.strip()


# ---------------------------------------------------------------------------
# Validation / heuristic repair
# ---------------------------------------------------------------------------

def validate_analysis_result(result: Dict[str, Any], content: str = "") -> Dict[str, Any]:
    """Validate and repair the analysis result.

    - If keywords is empty, extract capitalized words / nouns heuristically.
    - If context is empty, use the first sentence of content.
    - If tags is empty, derive from keywords.
    """
    if not isinstance(result, dict):
        result = {"keywords": [], "context": "", "tags": []}

    keywords = result.get("keywords", [])
    context = result.get("context", "")
    tags = result.get("tags", [])

    # Ensure lists
    if isinstance(keywords, str):
        keywords = _parse_list_items(keywords)
    if isinstance(tags, str):
        tags = _parse_list_items(tags)
    if isinstance(context, list):
        context = " ".join(context)

    # Repair empty keywords from content
    if not keywords and content:
        keywords = _heuristic_keywords(content)

    # Repair empty context from content
    if not context and content:
        context = _heuristic_context(content)

    # Repair empty tags from keywords
    if not tags and keywords:
        tags = keywords[:3]

    result["keywords"] = keywords
    result["context"] = context
    result["tags"] = tags
    return result


def _heuristic_keywords(content: str, max_keywords: int = 5) -> List[str]:
    """Extract heuristic keywords from content text."""
    # Remove common stop words and extract significant words
    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
        'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
        'as', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'out', 'off', 'over', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
        'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
        'if', 'while', 'about', 'up', 'it', 'its', 'i', 'me', 'my',
        'you', 'your', 'he', 'she', 'they', 'we', 'this', 'that', 'these',
        'those', 'what', 'which', 'who', 'whom', 'says', 'said', 'speaker',
    }
    words = re.findall(r'\b[a-zA-Z]{3,}\b', content)
    # Prefer capitalized words (likely proper nouns / key terms)
    scored = []
    seen = set()
    for w in words:
        w_lower = w.lower()
        if w_lower in stop_words or w_lower in seen:
            continue
        seen.add(w_lower)
        score = 2 if w[0].isupper() else 1
        scored.append((w_lower, score))

    scored.sort(key=lambda x: -x[1])
    return [w for w, _ in scored[:max_keywords]]


def _heuristic_context(content: str) -> str:
    """Extract a heuristic context sentence from content."""
    # Take the first sentence (up to period, question mark, or exclamation)
    match = re.match(r'(.+?[.!?])\s', content)
    if match:
        return match.group(1).strip()
    # Fallback: first 200 chars
    return content[:200].strip()
