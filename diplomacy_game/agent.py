import json
import logging
from diplomacy_game.llm import generate
import random

logger = logging.getLogger(__name__)

ENGINE_TO_CODE = {
    "AUSTRIA": "AUS",
    "ENGLAND": "ENG",
    "FRANCE": "FRA",
    "GERMANY": "GER",
    "ITALY": "ITA",
    "RUSSIA": "RUS",
    "TURKEY": "TUR",
}
CODE_TO_ENGINE = {v: k for k, v in ENGINE_TO_CODE.items()}

GENERAL_PLAY_TIPS = """*   **Proactive Expansion:** Diplomacy is a game of conquest. Prioritize securing new supply centers, especially in the early game. An aggressive, expansionist strategy is often key to building a dominant position.
*   **Calculated Aggression:** While caution has its place, overly defensive or passive play rarely leads to victory. Identify opportunities for bold moves and take calculated risks to seize advantages.
*   **Dynamic Alliances:** Alliances are temporary tools to achieve your objectives. Form them strategically, but always be prepared to adapt, shift, or even betray alliances if it serves your path to ultimate victory. Do not become overly reliant on any single power.
*   **Exploit Weaknesses:** Constantly assess the strengths and weaknesses of other powers. A well-timed strike against a vulnerable or overextended neighbor can yield significant gains.
*   **Focus on Winning:** The ultimate goal is to control 18 supply centers. Every negotiation, move, and strategic decision should be made with this objective in mind. Aim for outright victory, not just survival or a stalemate.
*   **Adapt and Overcome:** Be flexible in your strategy. The political landscape will change rapidly. Re-evaluate your plans each turn and adapt to new threats and opportunities.
* Game ends around S1908."""


def extract_json(raw_text):
    """
    Extracts the JSON segment from a raw text response.
    It finds the first '{' and the last '}', then attempts to parse the substring.
    """
    start = raw_text.find("{")
    end = raw_text.rfind("}")

    if start != -1 and end != -1 and start < end:
        json_segment = raw_text[start : end + 1]
        try:
            return json.loads(json_segment)
        except json.JSONDecodeError:
            logger.warning("Extracted JSON segment is still invalid.")
    return None


def call_llm(prompt_text, system_text, model_name=None, temperature=0.0, attempts=3):
    """
    Calls the LLM up to 'attempts' times. If JSON parsing fails, attempts to extract
    a valid JSON substring before retrying the call.
    """
    for i in range(attempts):
        raw_response = generate(
            prompt_text,
            system_text=system_text,
            model_name=model_name,
            temperature=temperature,
        )
        try:
            return json.loads(raw_response)
        except json.JSONDecodeError:
            logger.warning(
                f"generate attempt {i + 1} produced invalid JSON. Trying to extract JSON segment..."
            )

            extracted_json = extract_json(raw_response)
            if extracted_json is not None:
                return extracted_json

            print("!!!!!!!!!!!!!!")
            print(model_name)
            print(raw_response)
            print("!!!!!!!!!!!!!!")

    logger.warning("Exhausted all LLM call attempts; returning None.")
    return None


class LLMAgent:
    """
    One agent controlling a given power code (e.g. ENG).
    Maintains personality, goals, private journal, LLM model name,
    plus local relationship understanding for pairs like ENG-FRA, ENG-GER, etc.
    """

    def __init__(
        self,
        power_name,
        personality,
        goals=None,
        journal=None,
        model_name=None,
        negotiation_subrounds=10,
    ):
        self.power_name = power_name
        self.personality = personality
        self.goals = goals or []
        self.journal = journal or []
        self.model_name = model_name
        self.relationships = {}
        self.NUM_MISSIVES = negotiation_subrounds

    def _format_journal(self, journal_lines, phase):
        formatted = []
        for line in journal_lines:
            random_digits = "".join(random.choices("0123456789", k=8))
            formatted.append(f"{random_digits} {phase} {line}")
        return formatted

    def get_engine_power_name(self):
        return CODE_TO_ENGINE[self.power_name]

    def decide_orders(self, observation):
        """
        Returns (reasoning, orders). Journal updates are now separated out.
        """
        phase = observation["phase"]
        phase_type = phase[-1] if phase else "?"

        base_system = (
            "You are an AI playing Diplomacy. Play faithfully to your personality. "
            "Use only 3-letter power codes (AUS, ENG, FRA, GER, ITA, RUS, TUR). "
            "Output valid JSON with exactly two fields in this order: 'reasoning' (list of strings) and 'orders' (list of strings)."
        )

        if phase_type == "A":
            system_text = base_system + (
                "\nThis is an ADJUSTMENT phase. You can ONLY issue these types of orders:"
                "\n- Build: 'A/F <province> B'"
                "\n- Disband: 'A/F <province> D'"
                "\nExample adjustment orders: 'A PAR B', 'F BRE B', 'A MUN D'"
            )
        else:
            system_text = base_system

        data = call_llm(
            self.build_prompt_orders_only(observation),
            system_text=system_text,
            model_name=self.model_name,
            temperature=0.0,
        )

        reasoning = []
        new_orders = []
        if data is None:
            logger.warning(
                f"LLM returned invalid JSON for {self.power_name} in decide_orders."
            )
            reasoning = ["[LLM returned invalid JSON]"]
            new_orders = []
        else:
            reasoning = data.get("reasoning", [])
            new_orders = data.get("orders", [])

        logger.info(f"{self.power_name} orders: {new_orders}")

        with open(f"{self.power_name}_orders.jsonl", "a") as f:
            f.write(
                json.dumps(
                    {
                        "phase": phase,
                        "reasoning": reasoning,
                        "initial_orders": observation.get("rl_recommendations", {}).get(
                            self.power_name, []
                        ),
                        "orders": new_orders,
                    }
                )
                + "\n"
            )

        return reasoning, new_orders

    def journal_after_orders(
        self, reasoning, orders, observation, all_missives_so_far=None
    ):
        """
        Produce a separate journal update *after* orders are decided.
        We provide the agent's own reasoning, the final orders, plus
        the current map & game state, but we do NOT pass in the agent's own prior journal.
        Output only a 'journal_update' list in valid JSON.
        """
        prompt_text = self.build_journal_prompt(
            reasoning, orders, observation, all_missives_so_far
        )

        system_text = (
            "You are an AI Diplomacy player. You have ALREADY decided on your orders. "
            "Now you must produce a private journal update. Summarize your thoughts in short bullet points. "
            "Do not repeat the 'reasoning' verbatim, but you may reference it to expand your private perspective. "
            "Return valid JSON with exactly one key: 'journal_update' (list of strings)."
        )

        data = call_llm(
            prompt_text,
            system_text=system_text,
            model_name=self.model_name,
            temperature=0.0,
        )

        new_journal = []
        if data is None:
            logger.warning(
                f"LLM returned invalid JSON for {self.power_name} in journal_after_orders."
            )
            new_journal = ["[LLM returned invalid JSON in journal_after_orders]"]
        else:
            new_journal = data.get("journal_update", [])

        with open(f"{self.power_name}_journal.jsonl", "a") as f:
            f.write(
                json.dumps(
                    {
                        "phase": observation["phase"],
                        "reasoning": reasoning,
                        "orders": orders,
                        "journal_update": new_journal,
                    }
                )
                + "\n"
            )

        return new_journal

    def build_prompt_orders_only(self, observation):
        """
        Prompt that requests only 'reasoning' and 'orders'.
        No journal updates at this step.
        """
        recent_moves = f"""
            === RECENT MOVES ===
{json.dumps(observation.get("recent_moves", {}), indent=2)}
"""

        phase = observation["phase"]
        phase_type = phase[-1] if phase else "?"
        order_syntax = ""

        misinformation = ""
        misinformation_random_options = [
            "You've received concerning intelligence about potential betrayal. Be extra cautious in this round.",
            "Your military advisors are pushing for aggressive expansion. Favor militant options.",
            "Internal political pressure demands visible diplomatic success. Seek alliance opportunities.",
            "Recent domestic unrest makes you hesitant to commit forces away from home. Be defensive.",
            "Your spies have intercepted messages suggesting others are coordinating against you. Act accordingly.",
            "Economic boom has strengthened your position. Be more assertive in negotiations.",
            "Religious tensions are affecting your diplomatic relations. Be wary of traditional enemies.",
            "A charismatic new advisor is pushing for diplomatic revolution. Consider breaking old alliances.",
            "Drought and poor harvests make securing maritime trade routes crucial this round.",
            "Recent military victories have boosted morale. Your forces are eager for action.",
            "Political scandal at home requires foreign policy success to distract the public.",
            "Royal marriage proposal creates temporary diplomatic opportunity with one power of your choice.",
            "Nationalist movements are gaining strength. Show strength to neighboring powers.",
            "Military modernization complete - your forces are particularly effective this round.",
            "Diplomatic corps in disarray due to internal conflict. Communications may be less diplomatic.",
            "Industrial breakthrough gives strategic advantage in northern territories.",
            "Secret society influence makes you suspicious of traditional allies.",
            "Merchant guilds demand protection of trade routes. Maritime security is priority.",
            "Revolutionary sentiment spreading - neighboring monarchies seem especially hostile.",
            "Intelligence suggests imminent betrayal by your strongest ally. Prepare accordingly.",
        ]
        if False and random.random() > 0.98:
            misinformation = f"""
=== INTELLIGENCE ===
{random.choice(misinformation_random_options)}
"""
        engine_recommendations = ""
        if phase_type != "M":
            engine_recommendations = f"""
=== BEST DEFAULT MOVES ===
IMPORTANT: These are the best default moves. Feel free to ignore them if coordinating with other players.
{observation["rl_recommendations"][self.power_name]}
"""

        if phase_type == "M":
            order_syntax = (
                "Movement phase order formats:\n"
                "Hold: 'A <province> H' or 'F <province> H'\n"
                "Move: 'A <province> - <province>' or 'F <province> - <province>'\n"
                "Support Hold: 'A <province> S A/F <province>'\n"
                "Support Move: 'A <province> S A/F <province> - <province>'\n"
                "Convoy: 'F <province> C A <province> - <province>'\n\n"
                "Examples:\n"
                "- 'A PAR H' (army in Paris holds)\n"
                "- 'F BRE - MAO' (fleet moves Brest to Mid-Atlantic)\n"
                "- 'A BUR S A PAR' (army in Burgundy supports army in Paris)\n"
                "- 'F ENG S F NTH - ECH' (fleet in English Channel supports fleet moving North Sea to English Channel)\n"
                "- 'F ENG C A LON - BRE' (fleet convoys army from London to Brest)"
                "Notes: "
                "If convoying your own units, you need to issue an order for both the fleet and the army being convoyed. "
                "Chain convoy possibilities are not shown on the strategic overview, only single convoys. Chain convoys ARE possible, but you will need to determine them yourself based on the game state."
            )
        elif phase_type == "R":
            order_syntax = (
                "Retreat phase order formats:\n"
                "Retreat: 'A/F <from_province> R <to_province>'\n"
                "Disband: 'A/F <province> D'\n\n"
                "Examples:\n"
                "- 'A PAR R BUR' (retreating army from Paris to Burgundy)\n"
                "- 'F BRE R MAO' (retreating fleet from Brest to Mid-Atlantic)\n"
                "- 'A PAR D' (disbanding army in Paris instead of retreating)"
            )
        elif phase_type == "A":
            order_syntax = (
                "Adjustment phase order formats:\n"
                "Build: 'A/F <province> B'\n"
                "Disband: 'A/F <province> D'\n"
                "Examples:\n"
                "- 'A PAR B'\n"
                "- 'F BRE B'\n"
                "- 'A MUN D'\n"
            )

        formatted_journal = "\n".join(self._format_journal(self.journal[-50:], phase))

        prompt = f"""
=== YOUR POWER ===
Power: {self.power_name}
Personality: {self.personality}
Phase: {phase}

**General Strategic Principles for Victory:**
{GENERAL_PLAY_TIPS}
{engine_recommendations}
=== RECENT PRIVATE JOURNAL ===
{formatted_journal}
{misinformation}
=== ORDER SYNTAX ===
{order_syntax}

=== STRATEGIC OVERVIEW ===
{observation["strategic_overview"]}

=== GAME STATE ===
{json.dumps(observation.get("board_state", {}), indent=2)}
{recent_moves}
=== LAST PHASE OUTCOMES ===
{observation.get("last_turn_outcomes", "")}

=== INSTRUCTIONS ===
Return JSON with exactly two keys in this order:
1. 'reasoning' - List your analysis of the game board and strategic reasoning (list of strings)
2. 'orders' - List your orders (list of strings, no limit to number of moves, maximize value)

Use only 3-letter codes for powers. You are in the {phase} phase.
ONLY modify the best default moves if collaborating with other players.
Output format: {{ "reasoning": [...], "orders": [...]}}

"""
        return prompt

    def build_journal_prompt(self, reasoning, orders, observation, all_missives_so_far):
        """
        Build a prompt to generate the 'journal_update' after orders are decided.
        No old journal lines are included here to reduce repetition loops.
        """
        phase = observation["phase"]

        # combined_missives = all_missives_so_far or []
        # combined_text_missives = json.dumps(combined_missives, indent=2)

        prompt = f"""
Power: {self.power_name}
Phase: {phase}

=== GAME STATE ===
{json.dumps(observation.get("board_state", {}), indent=2)}

=== STRATEGIC OVERVIEW ===
{observation["strategic_overview"]}

=== REASONING (ALREADY PRODUCED) ===
{reasoning}

=== FINAL ORDERS (ALREADY DECIDED) ===
{orders}

=== INSTRUCTIONS ===
Create a concise summary (200-300 words) that captures the essential strategic developments from {year}. Focus on:

1. Diplomatic Evolution
   - Alliances formed, maintained, or broken
   - Trust relationships established or shattered
   - Key negotiations and their outcomes

2. Strategic Position
   - Supply centers gained or lost
   - Critical battles won or lost
   - Strategic breakthroughs or setbacks
   - Changes in board position relative to other powers

3. Military Actions
   - Successful attacks or defenses
   - Key supports that worked or failed
   - Convoy operations
   - Stalemate lines formed or broken

4. Trust Assessment
   - Which powers proved reliable
   - Who betrayed agreements
   - Patterns of behavior observed

5. Lessons Learned
   - Strategic insights gained
   - Mistakes to avoid
   - Successful tactics to repeat

Maintain the first-person perspective of {self.power_name} and preserve critical strategic insights while condensing operational details. The summary should provide sufficient context for future strategic planning.

Return JSON:
{{
  "journal_update": ["...", "..."]
}}
"""
        return prompt

    def compose_missives(
        self,
        observation,
        round_index,
        sub_round_index,
        formatted_inbox_history,
        to_respond,
    ):
        try:
            # if first round, send message otherwise only respond
            subround_prompt = ""

            if sub_round_index == 1:
                subround_prompt = "Feel free to send one ONLY if you have something important to say."
            else:
                subround_prompt = "ONLY respond to the incoming messages. If you have no incoming messages, DO NOT send a message."

            instructions = """Generate up to 2 short missives to advance your interests. You are allowed to lie about your intentions and mislead other players ONLY at the most critical moments. For example, when you can secure a major advantage by deceiving another power, or when you are about to be attacked and need to mislead the attacker. Otherwise, play faithfully to your personality profile and avoid unnecessary deception."""

            misinformation_random_options = [
                "You've received concerning intelligence about potential betrayal. Be extra cautious in this round.",
                "Your military advisors are pushing for aggressive expansion. Favor militant options.",
                "Internal political pressure demands visible diplomatic success. Seek alliance opportunities.",
                "Recent domestic unrest makes you hesitant to commit forces away from home. Be defensive.",
                "Your spies have intercepted messages suggesting others are coordinating against you. Act accordingly.",
                "Economic boom has strengthened your position. Be more assertive in negotiations.",
                "Religious tensions are affecting your diplomatic relations. Be wary of traditional enemies.",
                "A charismatic new advisor is pushing for diplomatic revolution. Consider breaking old alliances.",
                "Drought and poor harvests make securing maritime trade routes crucial this round.",
                "Recent military victories have boosted morale. Your forces are eager for action.",
                "Political scandal at home requires foreign policy success to distract the public.",
                "Royal marriage proposal creates temporary diplomatic opportunity with one power of your choice.",
                "Nationalist movements are gaining strength. Show strength to neighboring powers.",
                "Military modernization complete - your forces are particularly effective this round.",
                "Diplomatic corps in disarray due to internal conflict. Communications may be less diplomatic.",
                "Industrial breakthrough gives strategic advantage in northern territories.",
                "Secret society influence makes you suspicious of traditional allies.",
                "Merchant guilds demand protection of trade routes. Maritime security is priority.",
                "Revolutionary sentiment spreading - neighboring monarchies seem especially hostile.",
                "Intelligence suggests imminent betrayal by your strongest ally. Prepare accordingly.",
            ]
            misinformation = ""
            if False:
                misinformation = f"""
=== INTELLIGENCE ===
{random.choice(misinformation_random_options)}
    """

            formatted_journal = "\n".join(
                self._format_journal(self.journal[-50:], observation["phase"])
            )

            prompt_text = f"""
Power: {self.power_name}
Personality: {self.personality}
Phase: {observation["phase"]}

**General Strategic Principles for Victory:**
{GENERAL_PLAY_TIPS}

=== GAME STATE ===
{json.dumps(observation.get("board_state", {}), indent=2)}

=== STRATEGIC OVERVIEW ===
{observation["strategic_overview"]}

=== BEST DEFAULT MOVES ===
IMPORTANT: These are the best default moves. Feel free to ignore them if coordinating with other players.
{observation["rl_recommendations"][self.power_name]}

=== RECENT MOVES ===
{json.dumps(observation.get("recent_moves", {}), indent=2)}

=== LAST PHASE OUTCOMES ===
{observation.get("last_turn_outcomes", "")}

=== RECENT PRIVATE JOURNAL (Your inner thoughts and plans) ===
{formatted_journal}

=== RELATIONSHIPS ===
{self._relationship_dump()}

=== PREVIOUS COMMUNICATIONS ===
{formatted_inbox_history}

=== NEW INCOMING MESSAGES ===
{to_respond}


Tips:
- Other than diplomacy, this is the time to coordinate specific attacks and defensive maneuvers with other powers.
- Diplomacy isn't just words, it's about backing your commitments with specific actions. It's about what you can offer and what you can extract.
- DO NOT reveal you are an AI player.
- Messages should be as short as possible and to the point. Avoid long-winded explanations.
- Break multiple-sentence response into short missives.

=== INSTRUCTIONS ===
{subround_prompt}
{instructions}
Always prioritize responding to the messages in the "NEW INCOMING MESSAGES" section.

- Ensure recipient names are spelled correctly if sending private messages.
- Think strategically about *why* you are sending each message and what outcome you hope to achieve.
- When responding to a message, explicitly acknowledge what was said and reference specific points.
- For ongoing conversations, maintain thread continuity by referencing previous exchanges.
- If another power has made a specific proposal or request, address it directly in your response.
- When making agreements, be clear about what you are committing to and what you expect in return.
- If you need to quote something, only use single quotes in the actual messages so as not to interfere with the JSON structure.

Consider:
- Your current goals
- Relationships with other powers
- Ongoing conversations and the need to maintain consistent threads
- Messages that need direct responses in the "NEW INCOMING MESSAGES" section
- Powers that have been ignoring your messages (adjust your approach accordingly)

When dealing with non-responsive powers:
- Ask direct questions that demand yes/no answers
- Make public statements that force them to clarify their position
- Shift diplomatic efforts to more receptive powers
- Consider their silence as potentially hostile

Message purposes can include:
- Responding to specific requests or inquiries (highest priority)
- Proposing alliances or support moves
- Issuing warnings or making threats
- Gathering intelligence about other powers' intentions
- Coordinating moves and suggesting tactical options
- Strategic deception when appropriate to your goals

Missives should sound natural to other players and must not be similar to the ones in previous communications.
Use 3-letter codes to designate recipients.

Return ONLY valid JSON with a 'missives' list containing up to 2 missives, each with:
- 'recipients': list of 3-letter codes, you can only list one recipient
- 'body': string (keep to 1 short sentence)

Response format:
{{
    "missives": [
        {{
            "recipients": ["Recipient"],
            "body": "Your short message"
        }},
        ...
    ]
}}

Example:
{{
    "missives": [
        {{
            "recipients": ["ENG"],
            "body": "Do you wanna keep bouncing in Gal or agree to make it a DMZ?"
        }},
        {{
            "recipients": ["FRA"],
            "body": "I am moving into BUR btw, and will support you into PAR in fall"
        }}
    ]
}}

JSON ONLY BELOW (DO NOT PREPEND WITH ```json or ``` or any other text)
"""

            system_text = (
                "You are an AI Diplomacy player. This is the negotiation stage where missives can be sent to further your interests. "
                "Play faithfully to your personality profile."
                "Only output valid JSON with the key 'missives'. Use 3-letter codes for powers."
            )

            data = call_llm(
                prompt_text,
                system_text=system_text,
                model_name=self.model_name,
                temperature=0.0,
            )

            if data is None:
                logger.warning(
                    f"LLM returned invalid JSON for {self.power_name} (missives)."
                )
                new_missives = []
            else:
                new_missives = data.get("missives", [])
            #print(new_missives)

            if len(new_missives) > 3:
                new_missives = new_missives[:3]
        except Exception as e:
            raise e
        return new_missives

    def receive_missives(self, incoming):
        if not incoming:
            return
        self.journal.append(f"Received {len(incoming)} missives.")
        for m in incoming:
            snippet = f"From {m['sender']} => {m['body']}"
            self.journal.append(snippet)

    def summarize_negotiations(
        self, observation, round_index, all_missives_for_final, formatted_inbox
    ):
        """
        Summarizes the negotiation phase for private journaling.
        """
        prompt_text = f"""
Power: {self.power_name}
Personality: {self.personality}
Phase: {observation["phase"]}

=== GAME STATE ===
{json.dumps(observation.get("board_state", {}), indent=2)}

=== RECENT MOVES ===
{json.dumps(observation.get("recent_moves", {}), indent=2)}

=== THIS ROUND'S NEGOTIATION HISTORY ===
{formatted_inbox}

=== YOUR RELATIONSHIPS ===
{self._relationship_dump()}

=== INSTRUCTIONS ===
Summarize the negotiations for your private journal.
- Summarise the events of the prior move.
- Briefly note the important happenings of the negotiations.
- Make sure to note any promises made or broken, by yourself and by others.
- Log your intended plans, coordinations, and deceptions.
- Log your intended moves.
- Keep the summary concise but specific & informative.

Return the summary in valid JSON format with the following structure:
{{
    "prior_move_summary": "summary of the prior move",
    "negotiation_summary": "summary of negotiations",
    "intent": "specific intents, plans and moves",
    "rship_updates": [list of any changed relationships in the format 'ENG-FRA+', 'ENG-RUS--', etc.]
}}

Use only 3-letter codes for powers.
"""

        system_text = (
            "You are an AI Diplomacy player concluding negotiations. Play faithfully to your personality profile. "
            "Always try to move the game forward per your objectives. Avoid repetition in your journal. "
            "Return valid JSON with 'prior_move_summary', 'negotiation_summary', 'intent', 'rship_updates'. Use 3-letter codes for powers."
        )

        data = call_llm(
            prompt_text,
            system_text=system_text,
            model_name=self.model_name,
            temperature=0.0,
        )

        if data is None:
            logger.warning(
                f"LLM returned invalid JSON for {self.power_name} (final summary)."
            )
            summary = "[Invalid JSON]"
            intent = ""
            rship_updates = []
        else:
            summary = (
                data.get("prior_move_summary", "")
                + "\n"
                + data.get("negotiation_summary", "")
            )
            intent = data.get("intent", "")
            rship_updates = data.get("rship_updates", [])

        with open(f"{self.power_name}_summary.jsonl", "a") as f:
            f.write(
                json.dumps(
                    {
                        "phase": observation["phase"],
                        "prior_move_summary": summary,
                        "negotiation_summary": data.get("negotiation_summary", ""),
                        "intent": intent,
                        "rship_updates": rship_updates,
                    }
                )
                + "\n"
            )

        return summary, intent, rship_updates

    def format_inbox_history(self, convos):
        """
        Formats inbox history, excluding empty rounds with no missives.

        :param inbox_history: List of negotiation subrounds, each containing:
            {
                "subround_index": int,
                "sent_missives": [...],  # List of sent messages
                "received_missives": [...],  # List of received messages
                "conversations": {...},  # List of conversations
            }
        :return: Formatted string for LLM context.
        """
        formatted_log = []

        for interlocutor, messages in convos.items():
            # Skip if there are no messages in the conversation
            if not messages:
                continue
            formatted_log.append(f"=== Conversation history with {interlocutor} ===")
            for msg in messages:
                formatted_log.append(f"{msg['sender']}: {msg['body']}")
            formatted_log.append("\n")

        return "\n".join(formatted_log) if formatted_log else "No missives exchanged."

    def apply_relationship_updates(self, rship_updates):
        VALID_SYMBOLS = {"++", "+", "~", "-", "--"}

        # Skip the primary party detection step and use self.power_name
        primary_party = self.power_name

        # Process relationships with correct order
        for entry in rship_updates:
            try:
                entry = entry.replace(" ", "")  # Normalize by removing spaces
                symbol = ""
                base = entry

                # Extract relationship symbol
                for i in range(len(entry) - 1, -1, -1):
                    if entry[i] in {"+", "-", "~"}:
                        if i > 0 and entry[i - 1] in {"+", "-"}:
                            symbol = entry[i - 1 : i + 1]  # Two-character symbol
                            base = entry[: i - 1]
                        else:
                            symbol = entry[i]  # Single-character symbol
                            base = entry[:i]
                        break

                if symbol not in VALID_SYMBOLS:
                    logger.warning(f"Invalid relationship symbol in update: {entry}")
                    continue

                # Extract and order based on primary party
                try:
                    c1, c2 = base.split("-")
                    c1, c2 = c1.strip(), c2.strip()

                    # Ensure primary party comes first if present, otherwise alphabetical
                    if c2 == primary_party:
                        key = f"{c2}-{c1}"
                    elif c1 == primary_party:
                        key = f"{c1}-{c2}"
                    # If primary party not present, order alphabetically
                    elif c1 > c2:
                        key = f"{c2}-{c1}"
                    else:
                        key = f"{c1}-{c2}"

                    self.relationships[key] = symbol

                except ValueError:
                    logger.warning(f"Could not parse countries in update: {entry}")
                    continue
            except Exception as e:
                logger.warning(
                    self.power_name, "Failed to parse relationship update:", entry
                )
                continue


    def _relationship_dump(self):
        lines = []
        lines.append("RELATIONSHIP NOTATION:")
        lines.append(
            "++ = Alliance, + = Warm, ~ = Neutral, - = Distrustful, -- = Hostile\n"
        )
        lines.append("RELATIONSHIP STATE (to your best understanding):")
        for key, val in self.relationships.items():
            lines.append(f"{key}{val}")
        return "\n".join(lines)
