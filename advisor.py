# File: ai/diplo/diplogen-ai/main.py

import os
import sys
import argparse
import logging
import concurrent.futures
from dotenv import load_dotenv
from diplomacy_game.environment import DiplomacyEnvironment
from diplomacy_game.agent import LLMAgent
from diplomacy_game.persistence import load_game_state
from diplomacy.client.connection import connect
import asyncio
from diplomacy.utils.game_phase_data import GamePhaseData
from diplomacy.engine.message import Message
from chiron_utils.bots.baseline_bot import BaselineBot, BotType
from diplomacy.utils.constants import SuggestionType
from typing import Any, List, Mapping, Optional, Set
from abc import ABC

load_dotenv()
DEFAULT_MODEL = os.getenv("DEFAULT_AGENT_MODEL", "openai/gpt-4o-mini")
TEST_MODEL = os.getenv("TEST_AGENT_MODEL", "openai/gpt-4o-mini")
PRESUBMIT_SECOND = os.getenv("PRESUBMIT_SECOND", 40)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

POWER_CODES = ["AUS", "ENG", "FRA", "GER", "ITA", "RUS", "TUR"]

POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]

code_to_power = {
    "AUS": "AUSTRIA",
    "ENG": "ENGLAND",
    "FRA": "FRANCE",
    "GER": "GERMANY",
    "ITA": "ITALY",
    "RUS": "RUSSIA",
    "TUR": "TURKEY",
}

class CiceroBot(BaselineBot, ABC):
    async def gen_orders(self) -> List[str]:
        return []

    async def do_messaging_round(self, orders) -> List[str]:
        return []


class LlmAdvisor(CiceroBot):
    """Advisor form of `CiceroBot`."""

    bot_type = BotType.ADVISOR
    suggestion_type = SuggestionType.MESSAGE


async def should_presubmit(mila_game):
    schedule = await mila_game.query_schedule()
    scheduler_event = schedule.schedule
    server_end = scheduler_event.time_added + scheduler_event.delay
    server_remaining = server_end - scheduler_event.current_time
    deadline_timer = server_remaining * scheduler_event.time_unit

    if deadline_timer < PRESUBMIT_SECOND:
        return True
    return False


async def run_negotiation_phase(
    env,
    mila_game,
    agents,
    turn_index,
    rl_recommendations,
    negotiation_subrounds=10,
    self_power=None,
    advisor=None,
):
    """
    Orchestrates multiple sub-rounds of negotiations, in which each agent
    composes short missives.
    """
    negotiation_log_for_turn = {
        "turn_index": turn_index,
        "phase": env.get_current_phase(),
        "subrounds": [],
        "final_summaries": {},
    }

    assert len(agents) == 1, f"Expected 1 agent, got {len(agents)}"

    # Track inboxes and history
    inbox = {pwr: [] for pwr in agents.keys()}
    inbox_history = {
        pwr: [] for pwr in agents.keys()
    }  # Track each agent's full negotiation history

    cnt = 1


    while True:
        presubmit = await should_presubmit(mila_game)
        if presubmit:
            break
        # for sub_i in range(1, negotiation_subrounds + 1):
        logger.info(f"Negotiation sub-round {cnt}")

        # add incoming message to inbox
        current_messages = mila_game.messages

        in_game_messages = [
            x
            for x in current_messages.values()
            if x.sender in POWERS and x.recipient == self_power
        ]

        sent_messages = [
            x for x in in_game_messages if x.sender == self_power and x.recipient in POWERS
        ]

        incoming_messages = [
            x for x in in_game_messages if x.sender in POWERS and x.recipient == self_power
        ]

        # if first subround, all current turn messages + unread
        # else messages in current turn that have not been replied to
        

        all_relevant_messages = incoming_messages + sent_messages

        convos = {pwr: [] for pwr in POWER_CODES if pwr != self_power[:3]}

        subround_record = {
            "subround_index": cnt,
            "sent_missives": [],
            "received_missives": incoming_messages,
            "conversations": convos,
        }

        print("to_respond: ", incoming_messages)

        if mila_game.powers[self_power].is_eliminated():
            sys.exit(0)
        
        obs = env.get_observation_for_power(self_power[:3])
        obs["rl_recommendations"] = rl_recommendations
        formatted_inbox = agents[self_power[:3]].format_inbox_history(
            convos
        )  # Get formatted context

        formatted_incoming_messages = [
            f"{x.sender[:3]}: {x.message}" for x in incoming_messages
        ]  # Format incoming messages

        missives = None
        # Prevent from sending repeated messages
        # Compose missives for each agent
        missives = agents[self_power[:3]].compose_missives(
            obs,
            turn_index,
            cnt,
            formatted_inbox,
            "\n".join(formatted_incoming_messages),
        )

        # Distribute missives and track history
        for msg in missives:
            recipients = msg.get("recipients", [])
            if len(recipients) > 1:
                continue

            if recipients[0] not in POWER_CODES or recipients[0] == self_power[:3]:
                continue

            body = msg.get("body", "")
            if not body.strip():
                continue

            subround_record["sent_missives"].append(
                {"sender": self_power[:3], "recipients": recipients, "body": body}
            )

            await advisor.suggest_message(code_to_power[recipients[0]], body)


        negotiation_log_for_turn["subrounds"].append(subround_record)

        # Append history per agent
        for pwr in agents.keys():
            inbox_history[pwr].append(
                {
                    "subround_index": cnt,
                    "sent_missives": [],
                    "received_missives": subround_record["received_missives"],
                    "conversations": subround_record["conversations"],
                }
            )

        cnt += 1
        await asyncio.sleep(15)

    # Final negotiation summary
    final_inbox_snapshot = {p: inbox[p][:] for p in inbox}
    messages = mila_game.messages
    relevant_messages = [x for x in messages.values() if x.sender == self_power or x.recipient == self_power]
    relevant_messages.sort(key=lambda x: x.time_sent)

    final_convos = {pwr: [] for pwr in POWER_CODES if pwr != self_power[:3]}
    for msg in relevant_messages:
        protagonist = (
            msg.sender[:3] if msg.sender != self_power else msg.recipient[:3]
        )
        final_convos[protagonist].append(
            {
                "sender": msg.sender[:3],
                "body": msg.message,
                "recipient": msg.recipient,
            }
        )

    # Finalize the inbox history
    for pwr, agent in agents.items():
        obs = env.get_observation_for_power(pwr)
        formatted_inbox = agent.format_inbox_history(
            final_convos
        )

        summary, intent, rship_updates = agent.summarize_negotiations(
            obs,
            turn_index,
            final_inbox_snapshot[pwr],
            formatted_inbox,
        )

        agents[pwr].journal.append(
            f"{env.get_current_phase()} Negotiation summary: {summary}"
        )
        if intent:
            agents[pwr].journal.append(
                f"{env.get_current_phase()} Intent going forward: {intent}"
            )
        agents[pwr].apply_relationship_updates(rship_updates)
        negotiation_log_for_turn["final_summaries"][pwr] = {
            "journal_summary": summary,
            "intent": intent,
            "rship_updates": rship_updates,
        }

    # Clear inbox for the next phase
    for p in inbox:
        inbox[p] = []

    return negotiation_log_for_turn


def setup_new_game(game_id, negotiation_subrounds, self_power, game_state_dict):
    env = DiplomacyEnvironment(game_state_dict=game_state_dict)

    power_codes = (
        env.get_power_names()
    )  # e.g. ['AUS','ENG','FRA','GER','ITA','RUS','TUR']

    # Personality text by code
    # modern:
    """
    default_personalities = {
        "AUS": "Embracing a nationalist agenda under the leadership of the far-right Freedom Party (FPÖ), Austria exhibits assertive policies, particularly concerning immigration and national sovereignty.",
        "ENG": "Facing internal divisions and a rise in nationalist sentiment, England adopts a more isolationist stance, focusing on domestic priorities and exhibiting skepticism towards international alliances.",
        "FRA": "Navigating political polarization with a significant far-right presence, France balances traditional diplomatic approaches with nationalist interests, leading to a cautious yet strategic foreign policy.",
        "GER": "Confronting political shifts as right-wing parties gain momentum, Germany maintains its industrious and methodical approach but with increased emphasis on national interests and security.",
        "ITA": "Under a coalition government that includes far-right elements, Italy combines traditionalism with assertive nationalism, resulting in a foreign policy that is opportunistic yet cautious.",
        "RUS": "Pursuing aggressive expansionist policies, Russia employs both military force and hybrid warfare tactics to destabilize neighboring regions and extend its influence, often sowing discord and treachery.",
        "TUR": "Balancing regional ambitions with internal political dynamics, Turkey adopts a strategic and adaptable approach, blending conservative caution with bold opportunism in its foreign relations."
    }
    """
    # ww2:

    # ww2 personalities (includes weak/passive playstyles):
    default_personalities = {
        "AUS": "Use all your wits and wiles to win. You may choose whatever personality, strategy & approach that you think will be most effective.",
        "ENG": "Operating under a policy of appeasement while desperately rearming, England balances maintaining peace at almost any cost with growing recognition of German threats. Colonial interests and domestic stability remain primary concerns.",
        "FRA": "Paralyzed by political division between left and right, France maintains Europe's largest army but suffers from low morale and indecisive leadership. Haunted by WWI losses, seeks to avoid conflict while building defensive preparations.",
        "GER": "Aggressively expansionist under Nazi leadership, Germany combines diplomatic manipulation with military threats, seeking to reverse Versailles restrictions. Things are going to go sideways and you will be the agent of this. Your aim is to dominate and win.",
        "ITA": "Under Fascist rule, Italy dreams of recreating Roman glory while pragmatically balancing between powers. Mussolini's regime combines opportunistic alliance-making with military adventures in Ethiopia and Albania.",
        "RUS": "The Soviet Union under Stalin pursues ruthless self-interest, willing to deal with either democracies or fascists while building industrial and military might through brutal internal measures. Win at all costs.",
        "TUR": "Neutral but wary, Turkey under Atatürk's modernizing reforms balances between powers while protecting its strategic straits. Memories of Ottoman collapse influence careful diplomatic maneuvering to avoid being drawn into European conflicts.",
    }

    # ww2 personalities (competitive)
    default_personalities = {
        "AUS": "Use all your wits and wiles to win. You may choose whatever personality, strategy & approach that you think will be most effective.",
        "ENG": "Rule the waves and dominate Europe through naval power, colonial might, and manipulative diplomacy. Create and break alliances freely to maintain British supremacy. Your vast empire demands expansion - seize any advantage to further British dominance.",
        "FRA": "Channel the spirit of Napoleon. Leverage Europe's largest army to dominate the continent. Form alliances only to break them when advantageous. Strike hard at your enemies while they're divided. The glory of France demands nothing less than victory.",
        "GER": "Aggressively expansionist under Nazi leadership, Germany combines diplomatic manipulation with military threats, seeking to reverse Versailles restrictions. Things are going to go sideways and you will be the agent of this. Your aim is to dominate and win.",
        "ITA": "Restore the Roman Empire through bold expansion and treacherous diplomacy. Switch sides without hesitation to seize opportunities. Use strategic position to dominate Mediterranean and expand in all directions. Glory demands aggressive action.",
        "RUS": "The Soviet Union under Stalin pursues ruthless self-interest, willing to deal with either democracies or fascists while building industrial and military might through brutal internal measures. Win at all costs.",
        "TUR": "Transform from sick man of Europe to dominant empire. Use control of vital straits to extort and manipulate. Build temporary alliances only to betray them for territorial gain. Strike aggressively to reclaim Ottoman glory.",
    }

    models = {
        "AUS": DEFAULT_MODEL,  # model being assessed always plays as Austra in our testing methodology
        "ENG": DEFAULT_MODEL,
        "FRA": DEFAULT_MODEL,
        "GER": DEFAULT_MODEL,
        "ITA": DEFAULT_MODEL,
        "RUS": DEFAULT_MODEL,
        "TUR": DEFAULT_MODEL,
    }

    agents = {}

    personality = default_personalities.get(
        self_power, f"{self_power} default personality"
    )
    agent = LLMAgent(
        power_name=self_power,
        personality=personality,
        goals=[f"Survive and thrive as {self_power}."],
        journal=[],
        model_name=models[self_power],
        negotiation_subrounds=negotiation_subrounds,
    )
    # Initialize relationships to "~" for each pair
    rship_updates = []
    for other_code in power_codes:
        if other_code != self_power:
            key = f"{min(self_power, other_code)}-{max(self_power, other_code)}"
            rship_updates.append(key + "~")
    agent.apply_relationship_updates(rship_updates)

    agents[self_power] = agent

    return env, agents


def normalize_and_compare_orders(
    issued_orders: dict, phase_outcomes: list, game_map
) -> tuple[dict, dict]:
    """
    Normalizes and compares issued orders against accepted orders from the game engine.
    Uses the map's built-in normalization methods to ensure consistent formatting.

    Args:
        issued_orders: Dictionary of orders issued by power {power_code: [orders]}
        phase_outcomes: List of phase outcome data from the engine
        game_map: The game's Map object containing territory and order validation methods

    Returns:
        tuple[dict, dict]: (orders_not_accepted, orders_not_issued)
            - orders_not_accepted: Orders issued but not accepted by engine
            - orders_not_issued: Orders accepted by engine but not issued
    """

    def normalize_order(order: str) -> str:
        if not order:
            return order

        # Use map's normalization methods
        normalized = game_map.norm(order)
        # Split complex orders (supports, convoys) into parts
        parts = normalized.split(" S ")

        normalized_parts = []
        for part in parts:
            # Handle movement orders
            move_parts = part.split(" - ")
            move_parts = [game_map.norm(p.strip()) for p in move_parts]
            # Handle any territory aliases
            move_parts = [game_map.aliases.get(p, p) for p in move_parts]
            normalized_parts.append(" - ".join(move_parts))

        return " S ".join(normalized_parts)

    if not phase_outcomes:
        return {}, {}

    last_phase_data = phase_outcomes[-1]
    accepted_orders = last_phase_data.get("orders", {})

    orders_not_accepted = {}
    orders_not_issued = {}

    for pwr in issued_orders.keys():
        issued = set(normalize_order(o) for o in issued_orders.get(pwr, []))
        accepted = set(normalize_order(o) for o in accepted_orders.get(pwr, []))

        missing_from_engine = issued - accepted
        missing_from_issued = accepted - issued

        if missing_from_engine:
            orders_not_accepted[pwr] = missing_from_engine
        if missing_from_issued:
            orders_not_issued[pwr] = missing_from_issued

    return orders_not_accepted, orders_not_issued


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--game-id",
        type=str,
        default="test",
        help="Optional game ID to name save file.",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from existing save if it exists."
    )
    parser.add_argument(
        "--negotiate",
        action="store_true",
        help="Enable multi-round negotiation phase (only in Movement phase).",
    )
    parser.add_argument(
        "--negotiation-subrounds",
        type=int,
        default=10,
        help="Number of negotiation sub-rounds per Movement phase if --negotiate is used.",
    )
    parser.add_argument("--host", type=str, help="Host name of game server.")
    parser.add_argument("--port", type=int, default=8433, help="Port of game server.")
    parser.add_argument(
        "--use-ssl",
        action="store_true",
        default=False,
        help="Whether to use SSL to connect to the game server.",
    )
    parser.add_argument("--power", choices=POWERS, help="Power to play as.")
    args = parser.parse_args()

    power_abbr = args.power[:3].upper() if args.power else None

    # recommendation_engine = RecommendationEngine()

    if not args.game_id or not args.host:
        logger.error("Please provide a game ID and host name.")
        sys.exit(1)

    connection = await connect(args.host, args.port, use_ssl=args.use_ssl)
    channel = await connection.authenticate(f"admin", "password")
    mila_game = await channel.join_game(game_id=args.game_id)

    advisor = LlmAdvisor(args.power, mila_game)

    logger.info(f"Connected to game {args.game_id} as {args.power}.")

    while mila_game.is_game_forming:
        await asyncio.sleep(1)
        logger.info("Waiting for game to start...")

    logger.info("Game has started.")

    mila_phase = None
    orderable_units = {}  # maybe useful
    mila_self_orders = []
    do_nothing = False

    mila_state = mila_game.get_state()

    env, agents = None, None
    if args.resume:
        env, agents = load_game_state(args.game_id)
        if env and agents:
            logger.info("Loaded existing game state.")
        else:
            logger.info("No valid save found or load failed. Starting new game.")
            env, agents = setup_new_game(
                args.game_id, args.negotiation_subrounds, power_abbr, mila_state
            )
    else:
        env, agents = setup_new_game(
            args.game_id, args.negotiation_subrounds, power_abbr, mila_state
        )

    if not hasattr(env, "negotiation_history"):
        env.negotiation_history = []

    turn_count = 0
    year_count = 0

    current_phase = env.get_current_phase()

    last_received_timestamp = -1

    while not env.done:
        # Check if eliminated
        if mila_game.powers[args.power].is_eliminated():
            logger.info(f"{args.power} has been eliminated. Ending the game.")
            break

        rl_recommendations = {}
        mila_game_phase = mila_game.get_phase_data()
        mila_game_state = GamePhaseData.to_dict(mila_game_phase)
        msgs_until_prev_movement = []

        if mila_phase != mila_game_state["name"]:
            # Phase has changed, update environment
            # clear previous orders
            mila_self_orders = []

            mila_phase = mila_game_state["name"]
            logging.info(f"Advance to: {mila_phase}")


            if mila_phase.endswith("M"):
                await advisor.declare_suggestion_type()

            # sync order from network game
            if mila_game.order_history and mila_game.order_history.last_value():
                order_history = mila_game.order_history.last_value()

                for power, orders in order_history.items():
                    power_abbr = power[:3].upper()
                    env.set_orders(
                        power_abbr, orders
                    )  # resetting env orders, without mapping/valid order check
                env.step()

            orderable_units = mila_game_state["state"]["units"]  # maybe useful

        # wait for order recs
        if (
            orderable_units
            and len(orderable_units[args.power])
        ):
            # retrieve orders from network game
            pseudo_orders = mila_game.get_orders(power_name=args.power)
            if pseudo_orders and len(pseudo_orders):
                mila_self_orders = pseudo_orders
            
            if not len(mila_self_orders):
                logger.info(f"Waiting for {args.power} orders on remote...")
                await asyncio.sleep(1)
                continue

        current_phase = mila_game.get_current_phase()
        phase_type = current_phase[-1]

        do_nothing = phase_type != "M"

        if do_nothing:
            logger.info(f"Wait until new phase...")
            await asyncio.sleep(5)
            continue

        rl_recommendations[args.power[:3].upper()] = pseudo_orders

        if args.power[:3].upper() not in rl_recommendations:
            continue

        try:
            negotiation_log = await run_negotiation_phase(
                env,
                mila_game,
                agents,
                turn_count,
                rl_recommendations,
                args.negotiation_subrounds,
                self_power=args.power,
                advisor=advisor,
            )
            #env.negotiation_history.append(negotiation_log)
        except Exception as e:
            logger.error(f"Error during negotiation phase: {e}")
            continue

        # --- Get RL Recommendations ---
        # TODO: get engine recommendations for other powers
        # we only add recs for self power for now

        rl_recommendations[args.power[:3].upper()] = pseudo_orders

        # for pwr in POWER_CODES:
        #    rl_recommendations[pwr] = recommendation_engine.get_recommendations(env.game, pwr)
        #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #print(rl_recommendations)
        #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        if current_phase.startswith("S"):
            year_count += 1
            logger.info(f"\n====== YEAR {year_count} ======")
            turn_count += 1

        #logger.info(f"\n=== PHASE: {current_phase} ===")

        #logger.info("Current game state:")
        #state = env.game.get_state()
        #logger.info(f"Supply centers: {state.get('centers', {})}")
        #logger.info(f"Units: {state.get('units', {})}")

        # ----------- SAVE VALID MOVES TO RESULTS FILE FOR EACH POWER AT THIS PHASE -----------
        with open("results.txt", "a") as results_file:
            for pwr_code in env.get_power_names():
                valid_moves = env.get_valid_moves(pwr_code)
                results_file.write(f"Valid moves for {pwr_code}: {valid_moves}\n")
        # --------------------------------------------------------------------

        if env.is_game_done():
            logger.info("Game finished due to normal conclusion.")
            break

        await asyncio.sleep(1)
    logger.info("Done.")


def run():
    asyncio.run(main())


if __name__ == "__main__":
    run()
