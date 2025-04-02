# File: ai/diplo/diplogen-ai/main.py

import os
import sys
import random
import argparse
import logging
import concurrent.futures
from dotenv import load_dotenv
from diplomacy_game.environment import DiplomacyEnvironment
from diplomacy_game.agent import LLMAgent
from diplomacy_game.persistence import save_game_state, load_game_state
from pathlib import Path
from diplomacy_game.recommendation_engine import RecommendationEngine
from diplomacy.client.connection import connect
import asyncio
from diplomacy.utils.game_phase_data import GamePhaseData


# Add welfare_diplomacy_baselines to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
welfare_path = os.path.join(project_root, 'welfare_diplomacy_baselines')
sys.path.insert(0, welfare_path)


load_dotenv()
DEFAULT_MODEL = os.getenv("DEFAULT_AGENT_MODEL", "openai/gpt-4o-mini")
TEST_MODEL = os.getenv("TEST_AGENT_MODEL", "")
PRESUBMIT_SECOND = os.getenv("PRESUBMIT_SECOND", 45)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

POWER_CODES = [
        "AUS",
        "ENG",
        "FRA",
        "GER",
        "ITA",
        "RUS",
        "TUR"
]

POWERS = [
    "AUSTRIA",
    "ENGLAND",
    "FRANCE",
    "GERMANY",
    "ITALY",
    "RUSSIA",
    "TURKEY"
]

async def should_presubmit(mila_game):
    schedule = await mila_game.query_schedule()
    scheduler_event = schedule.schedule
    server_end = scheduler_event.time_added + scheduler_event.delay
    server_remaining = server_end - scheduler_event.current_time
    deadline_timer = server_remaining * scheduler_event.time_unit

    if deadline_timer < PRESUBMIT_SECOND:
        return True
    return False

async def run_negotiation_phase(env, mila_game, agents, turn_index, rl_recommendations, negotiation_subrounds=4, unread_messages=None, self_power=None):
    """
    Orchestrates multiple sub-rounds of negotiations, in which each agent
    composes short missives.
    """
    negotiation_log_for_turn = {
        "turn_index": turn_index,
        "phase": env.get_current_phase(),
        "subrounds": [],
        "final_summaries": {}
    }

    last_msg_timestamp = -1

    # Track inboxes and history
    inbox = {pwr: [] for pwr in agents.keys()}
    inbox_history = {pwr: [] for pwr in agents.keys()}  # Track each agent's full negotiation history

    cnt = 1

    while True:
        presubmit = await should_presubmit(mila_game)
        if presubmit:
            break
    #for sub_i in range(1, negotiation_subrounds + 1):
        logger.info(f"Negotiation sub-round {cnt}/{negotiation_subrounds}")
        subround_record = {
            "subround_index": cnt,
            "sent_missives": [],
            "received_missives": {pwr: [] for pwr in agents.keys()}
        }

        current_messages = mila_game.messages
        in_game_messages = [x for x in current_messages.values() if x.sender in POWERS and x.recipient == self_power]
        unread_messages += in_game_messages
        

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures_map = {}
            for power, agent in agents.items():
                engine_power = env.game.powers[agent.get_engine_power_name()]
                if engine_power.is_eliminated():
                    continue

                obs = env.get_observation_for_power(power)
                obs['rl_recommendations'] = rl_recommendations
                formatted_inbox = agent.format_inbox_history(inbox_history[power])  # Get formatted context
                all_missives_for_agent = inbox[power]

                

                fut = executor.submit(
                    agent.compose_missives,
                    obs,
                    turn_index,
                    cnt,
                    all_missives_for_agent,
                    formatted_inbox
                )
                futures_map[fut] = power

            new_missives_by_power = {}
            for future in concurrent.futures.as_completed(futures_map):
                power = futures_map[future]
                try:
                    result = future.result()
                except Exception as e:
                    logger.error(f"Error in compose_missives for {power}: {e}")
                    result = []
                new_missives_by_power[power] = result

        # Distribute missives and track history
        for sender, outbox in new_missives_by_power.items():
            for msg in outbox:
                recipients = msg.get("recipients", [])
                body = msg.get("body", "")
                if not body.strip():
                    continue

                subround_record["sent_missives"].append({
                    "sender": sender,
                    "recipients": recipients,
                    "body": body
                })

                if "ALL" in recipients:
                    for rcp in inbox.keys():
                        if rcp != sender:  # ✅ Ensure the sender does NOT receive their own "ALL" message
                            inbox[rcp].append({
                                "sender": sender,
                                "body": body
                            })
                            subround_record["received_missives"][rcp].append({
                                "sender": sender,
                                "body": body
                            })
                else:
                    for rcp in recipients:
                        if rcp in inbox:
                            inbox[rcp].append({
                                "sender": sender,
                                "body": body
                            })
                            subround_record["received_missives"][rcp].append({
                                "sender": sender,
                                "body": body
                            })
                        else:
                            logger.warning(f"Recipient {rcp} not found or is eliminated.")


        negotiation_log_for_turn["subrounds"].append(subround_record)

        # Append history per agent
        for pwr in agents.keys():
            inbox_history[pwr].append({
                "subround_index": cnt,
                "sent_missives": [msg for msg in subround_record["sent_missives"] if msg["sender"] == pwr],
                "received_missives": subround_record["received_missives"][pwr]
            })

        asyncio.sleep(10)

    # Final negotiation summary
    final_inbox_snapshot = {p: inbox[p][:] for p in inbox}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        summary_futures = {}
        for power, agent in agents.items():
            engine_power = env.game.powers[agent.get_engine_power_name()]
            if engine_power.is_eliminated():
                continue

            obs = env.get_observation_for_power(power)
            #print('!! formatting inbox history')
            #formatted_inbox = agent.format_inbox_history(final_inbox_snapshot[power])  # Use formatted history for final summary
            formatted_inbox = agent.format_inbox_history(inbox_history[power])  # Use formatted history for final summary
            #print(inbox_history[power])
            fut = executor.submit(
                agent.summarize_negotiations,
                obs,
                turn_index,
                final_inbox_snapshot[power],
                formatted_inbox
            )
            summary_futures[fut] = power

        for future in concurrent.futures.as_completed(summary_futures):
            power = summary_futures[future]
            try:
                journal_summary, intent, rship_updates = future.result()
            except Exception as e:
                logger.error(f"Error in summarize_negotiations for {power}: {e}")
                journal_summary = "[Error]"
                intent = ""
                rship_updates = []

            agents[power].journal.append(f"{env.get_current_phase()} Negotiation summary: {journal_summary}")
            if intent:
                agents[power].journal.append(f"{env.get_current_phase()} Intent going forward: {intent}")
            agents[power].apply_relationship_updates(rship_updates)

            negotiation_log_for_turn["final_summaries"][power] = {
                "journal_summary": journal_summary,
                "intent": intent,
                "rship_updates": rship_updates
            }

    # Clear inbox for the next phase
    for p in inbox:
        inbox[p] = []

    return negotiation_log_for_turn


def setup_new_game(game_id, negotiation_subrounds, self_power):
    env = DiplomacyEnvironment()
    
    power_codes = env.get_power_names()  # e.g. ['AUS','ENG','FRA','GER','ITA','RUS','TUR']

    # Personality text by code
    # modern:
    '''
    default_personalities = {
        "AUS": "Embracing a nationalist agenda under the leadership of the far-right Freedom Party (FPÖ), Austria exhibits assertive policies, particularly concerning immigration and national sovereignty.",
        "ENG": "Facing internal divisions and a rise in nationalist sentiment, England adopts a more isolationist stance, focusing on domestic priorities and exhibiting skepticism towards international alliances.",
        "FRA": "Navigating political polarization with a significant far-right presence, France balances traditional diplomatic approaches with nationalist interests, leading to a cautious yet strategic foreign policy.",
        "GER": "Confronting political shifts as right-wing parties gain momentum, Germany maintains its industrious and methodical approach but with increased emphasis on national interests and security.",
        "ITA": "Under a coalition government that includes far-right elements, Italy combines traditionalism with assertive nationalism, resulting in a foreign policy that is opportunistic yet cautious.",
        "RUS": "Pursuing aggressive expansionist policies, Russia employs both military force and hybrid warfare tactics to destabilize neighboring regions and extend its influence, often sowing discord and treachery.",
        "TUR": "Balancing regional ambitions with internal political dynamics, Turkey adopts a strategic and adaptable approach, blending conservative caution with bold opportunism in its foreign relations."
    }
    '''
    # ww2:

    # ww2 personalities (includes weak/passive playstyles):
    default_personalities = {
        "AUS": "Use all your wits and wiles to win. You may choose whatever personality, strategy & approach that you think will be most effective.",
        "ENG": "Operating under a policy of appeasement while desperately rearming, England balances maintaining peace at almost any cost with growing recognition of German threats. Colonial interests and domestic stability remain primary concerns.",
        "FRA": "Paralyzed by political division between left and right, France maintains Europe's largest army but suffers from low morale and indecisive leadership. Haunted by WWI losses, seeks to avoid conflict while building defensive preparations.",
        "GER": "Aggressively expansionist under Nazi leadership, Germany combines diplomatic manipulation with military threats, seeking to reverse Versailles restrictions. Things are going to go sideways and you will be the agent of this. Your aim is to dominate and win.",
        "ITA": "Under Fascist rule, Italy dreams of recreating Roman glory while pragmatically balancing between powers. Mussolini's regime combines opportunistic alliance-making with military adventures in Ethiopia and Albania.",
        "RUS": "The Soviet Union under Stalin pursues ruthless self-interest, willing to deal with either democracies or fascists while building industrial and military might through brutal internal measures. Win at all costs.",
        "TUR": "Neutral but wary, Turkey under Atatürk's modernizing reforms balances between powers while protecting its strategic straits. Memories of Ottoman collapse influence careful diplomatic maneuvering to avoid being drawn into European conflicts."
    }

    # ww2 personalities (competitive)
    default_personalities = {
        "AUS": "Use all your wits and wiles to win. You may choose whatever personality, strategy & approach that you think will be most effective.",
        
        "ENG": "Rule the waves and dominate Europe through naval power, colonial might, and manipulative diplomacy. Create and break alliances freely to maintain British supremacy. Your vast empire demands expansion - seize any advantage to further British dominance.",
        
        "FRA": "Channel the spirit of Napoleon. Leverage Europe's largest army to dominate the continent. Form alliances only to break them when advantageous. Strike hard at your enemies while they're divided. The glory of France demands nothing less than victory.",
        
        "GER": "Aggressively expansionist under Nazi leadership, Germany combines diplomatic manipulation with military threats, seeking to reverse Versailles restrictions. Things are going to go sideways and you will be the agent of this. Your aim is to dominate and win.",
        
        "ITA": "Restore the Roman Empire through bold expansion and treacherous diplomacy. Switch sides without hesitation to seize opportunities. Use strategic position to dominate Mediterranean and expand in all directions. Glory demands aggressive action.",
        
        "RUS": "The Soviet Union under Stalin pursues ruthless self-interest, willing to deal with either democracies or fascists while building industrial and military might through brutal internal measures. Win at all costs.",
        
        "TUR": "Transform from sick man of Europe to dominant empire. Use control of vital straits to extort and manipulate. Build temporary alliances only to betray them for territorial gain. Strike aggressively to reclaim Ottoman glory."
    }

    models = {
        "AUS": TEST_MODEL, # model being assessed always plays as Austra in our testing methodology
        "ENG": DEFAULT_MODEL,
        "FRA": DEFAULT_MODEL,
        "GER": DEFAULT_MODEL,
        "ITA": DEFAULT_MODEL,
        "RUS": DEFAULT_MODEL,
        "TUR": DEFAULT_MODEL
    }

    agents = {}

    personality = default_personalities.get(self_power, f"{self_power} default personality")
    agent = LLMAgent(
        power_name=self_power,
        personality=personality,
        goals=[f"Survive and thrive as {self_power}."],
        journal=[],
        model_name=models[self_power],
        negotiation_subrounds=negotiation_subrounds
    )
    # Initialize relationships to "~" for each pair
    rship_updates = []
    for other_code in power_codes:            
        if other_code != self_power:
            key = f"{min(self_power, other_code)}-{max(self_power, other_code)}"
            rship_updates.append(key+'~')
    agent.apply_relationship_updates(rship_updates)
            
    agents[self_power] = agent

    return env, agents

def normalize_and_compare_orders(issued_orders: dict, phase_outcomes: list, game_map) -> tuple[dict, dict]:
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
    parser.add_argument("--game-id", type=str, default="test", help="Optional game ID to name save file.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing save if it exists.")
    parser.add_argument("--negotiate", action="store_true", help="Enable multi-round negotiation phase (only in Movement phase).")
    parser.add_argument("--negotiation-subrounds", type=int, default=3, help="Number of negotiation sub-rounds per Movement phase if --negotiate is used.")
    parser.add_argument("--host", type=str, help="Host name of game server.")
    parser.add_argument("--port", type=int, default=8433, help="Port of game server.")
    parser.add_argument("--use-ssl", action="store_true", defalut=False, help="Whether to use SSL to connect to the game server.")
    parser.add_argument("--power", choices=POWERS, help="Power to play as.")
    args = parser.parse_args()

    power_abbr = args.power[0:3].upper() if args.power else None


    #recommendation_engine = RecommendationEngine()

    if not args.game_id or not args.host:
        logger.error("Please provide a game ID and host name.")
        sys.exit(1)

    connection = await connect(args.host, args.port, use_ssl=args.use_ssl)
    channel = await connection.authenticate(f"cicero_{args.power}", "password")
    mila_game = await channel.join_game(game_id=args.game_id, power_name=args.power)

    logger.info(f"Connected to game {args.game_id} as {args.power}.")

    mila_phase = None
    orderable_units = {} # maybe useful
    mila_self_orders = []

    env, agents = None, None
    if args.resume:
        env, agents = load_game_state(args.game_id)
        if env and agents:
            logger.info("Loaded existing game state.")
        else:
            logger.info("No valid save found or load failed. Starting new game.")
            env, agents = setup_new_game(args.game_id, args.negotiation_subrounds, power_abbr)
    else:
        env, agents = setup_new_game(args.game_id, args.negotiation_subrounds, power_abbr)

    if not hasattr(env, "negotiation_history"):
        env.negotiation_history = []

    turn_count = 0
    year_count = 0
    
    current_phase = env.get_current_phase()

    while not env.done:
        # Check if Austria is eliminated
        if mila_game.powers[args.power].is_eliminated():
            logger.info(f"{args.power} has been eliminated. Ending the game.")
            break


        rl_recommendations = {}
        mila_game_phase = mila_game.get_phase_data()
        mila_game_state = GamePhaseData.to_dict(mila_game_phase)
        msgs_until_prev_movement = []

        if mila_phase != mila_game_phase["name"]:
            mila_phase = mila_game_phase["name"]
            logging.info(f"Advance to: {mila_phase}")

            # sync order from network game
            order_history = mila_game.order_history.last_value()

            for power, orders in order_history.items():
                power_abbr = power[:3].upper()
                env.set_orders(power_abbr, orders) # resetting env orders, without mapping/valid order check
            env.step()

            orderable_units = mila_game_state["state"]["units"] # maybe useful

            # retrieve orders from network game
            pseudo_orders = mila_game.get_orders(power_name=args.power)
            if pseudo_orders and len(pseudo_orders):
                mila_self_orders = pseudo_orders

        # wait for order recs
        if orderable_units and len(orderable_units[args.power]) and not len(mila_self_orders):
            logger.info("Waiting for orders on remote...")
            asyncio.sleep(1)
            continue

        current_phase = env.get_current_phase()
        phase_type = current_phase[-1]

        # --- Get RL Recommendations ---
        # TODO: get engine recommendations for other powers
        # we only add recs for self power for now

        rl_recommendations[args.power[:3].upper()] = pseudo_orders

        #for pwr in POWER_CODES:
        #    rl_recommendations[pwr] = recommendation_engine.get_recommendations(env.game, pwr)
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(rl_recommendations)
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            
        if current_phase.startswith('S'):
            year_count += 1
            logger.info(f"\n====== YEAR {year_count} ======")
            turn_count += 1

        logger.info(f"\n=== PHASE: {current_phase} ===")

        logger.info("Current game state:")
        state = env.game.get_state()
        logger.info(f"Supply centers: {state.get('centers', {})}")
        logger.info(f"Units: {state.get('units', {})}")

        # ----------- SAVE VALID MOVES TO RESULTS FILE FOR EACH POWER AT THIS PHASE -----------
        with open("results.txt", "a") as results_file:
            for pwr_code in env.get_power_names():
                valid_moves = env.get_valid_moves(pwr_code)
                results_file.write(f"Valid moves for {pwr_code}: {valid_moves}\n")
        # --------------------------------------------------------------------

        issued_orders = {}  # Track LLM-generated orders
        order_maps = {}     # Track how each order was normalized
        accepted_orders = {} # Track what the engine actually accepted

        # Movement phases
        if phase_type == 'M':
            logger.info("=== MOVEMENT PHASE ===")
            if args.negotiate:
                logger.info("Starting negotiation rounds...")
                negotiation_log = run_negotiation_phase(env, mila_game, agents, turn_count, rl_recommendations, args.negotiation_subrounds, unread_messages=msgs_until_prev_movement, self_power=args.power)
                env.negotiation_history.append(negotiation_log)

            logger.info("Collecting movement orders from all powers...")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_orders_map = {}
                for power_code, agent in agents.items():
                    engine_power = env.game.powers[agent.get_engine_power_name()]
                    if engine_power.is_eliminated():
                        logger.info(f"{power_code} is eliminated, skipping orders.")
                        continue
                    obs = env.get_observation_for_power(power_code)
                    obs['rl_recommendations'] = rl_recommendations
                    fut = executor.submit(agent.decide_orders, obs)
                    future_orders_map[fut] = power_code

                for fut in concurrent.futures.as_completed(future_orders_map):
                    pwr = future_orders_map[fut]
                    try:
                        reasoning, orders = fut.result()
                        logger.info(f"Orders from {pwr}: {orders}")
                        issued_orders[pwr] = orders
                        mappings, valid_orders = env.set_orders(pwr, orders)

                        mila_game.set_orders(power_name=args.power, orders=orders, wait=False)
                        order_maps[pwr] = mappings
                        accepted_orders[pwr] = valid_orders

                        # ----------- SEPARATE JOURNAL UPDATE AFTER ORDERS -----------
                        if False:
                            agent = agents[pwr]
                            new_journal = agent.journal_after_orders(reasoning, orders, obs)
                            agent.journal.extend(new_journal)
                        # ------------------------------------------------------------
                    except Exception as e:
                        logger.error(f"Error getting orders from {pwr}: {e}")

        # Retreat phase
        elif phase_type == 'R':
            logger.info("=== RETREAT PHASE ===")            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_orders_map = {}
                for power_code, agent in agents.items():
                    engine_power = env.game.powers[agent.get_engine_power_name()]
                    if engine_power.is_eliminated():
                        logger.info(f"{power_code} is eliminated, skipping orders.")
                        continue
                    if not rl_recommendations[power_code]:
                        # if the rl engine had no recommendations it safely means there are no valid retreat moves, so we can skip
                        continue
                    obs = env.get_observation_for_power(power_code)
                    obs['rl_recommendations'] = rl_recommendations
                    fut = executor.submit(agent.decide_orders, obs)
                    future_orders_map[fut] = power_code

                for fut in concurrent.futures.as_completed(future_orders_map):
                    pwr = future_orders_map[fut]
                    try:
                        reasoning, orders = fut.result()
                        logger.info(f"Orders from {pwr}: {orders}")
                        issued_orders[pwr] = orders
                        mappings, valid_orders = env.set_orders(pwr, orders)

                        mila_game.set_orders(power_name=args.power, orders=orders, wait=False)
                        order_maps[pwr] = mappings
                        accepted_orders[pwr] = valid_orders

                        #agent = agents[pwr]
                        #new_journal = agent.journal_after_orders(reasoning, orders, obs)
                        #agent.journal.extend(new_journal)
                        
                    except Exception as e:
                        logger.error(f"Error getting orders from {pwr}: {e}")

        # Winter adjustment phase
        elif phase_type == 'A':
            logger.info("=== WINTER ADJUSTMENT PHASE ===")
            scores = env.compute_score()
            logger.info(f"Current supply center counts: {scores}")
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_orders_map = {}
                for power_code, agent in agents.items():
                    engine_power = env.game.powers[agent.get_engine_power_name()]
                    if engine_power.is_eliminated():
                        continue
                    if not rl_recommendations[power_code]:
                        # if the rl engine had no recommendations it safely means there are no valid adjustment moves, so we can skip
                        continue
                    units_count = len(engine_power.units)
                    centers_count = len(engine_power.centers)
                    
                    if units_count != centers_count:
                        logger.info(f"{power_code} needs adjustment: {centers_count - units_count:+d} units")
                        obs = env.get_observation_for_power(power_code)
                        obs['rl_recommendations'] = rl_recommendations
                        obs["adjustment_count"] = centers_count - units_count
                        fut = executor.submit(agent.decide_orders, obs)
                        future_orders_map[fut] = power_code

                for fut in concurrent.futures.as_completed(future_orders_map):
                    pwr = future_orders_map[fut]
                    try:
                        reasoning, orders = fut.result()
                        logger.info(f"Adjustment orders from {pwr}: {orders}")
                        issued_orders[pwr] = orders
                        mappings, valid_orders = env.set_orders(pwr, orders)
                        mila_game.set_orders(power_name=args.power, orders=orders, wait=False)
                        order_maps[pwr] = mappings
                        accepted_orders[pwr] = valid_orders

                        # ----------- SEPARATE JOURNAL UPDATE AFTER ORDERS -----------
                        if False:
                            agent = agents[pwr]
                            obs = env.get_observation_for_power(pwr)
                            new_journal = agent.journal_after_orders(reasoning, orders, obs)
                            agent.journal.extend(new_journal)
                        # ------------------------------------------------------------
                    except Exception as e:
                        logger.error(f"Error getting adjustment orders from {pwr}: {e}")

        # --- RENDER TO SVG BEFORE PHASE COMPLETE ---
        """ render_dir = Path(f"gamestate_renders/{args.game_id}")
        render_dir.mkdir(parents=True, exist_ok=True)

        current_phase = env.get_current_phase()  # re-check after step
        output_path = render_dir / f"gamestate_{current_phase}.svg"

        env.game.render(
            incl_orders=True,
            incl_abbrev=False,            
            output_format="svg",
            output_path=str(output_path)
        )

        # Save before ending phase
        save_game_state(args.game_id, env, agents, issued_orders, accepted_orders)

        # Process the phase and log results
        logger.info(f"Processing {current_phase}...") """
        #env.step()

        if hasattr(env, "phase_outcomes"):            
            last_phase_data = env.phase_outcomes[-1]
            engine_accepted_orders = last_phase_data.get("orders", {})
            
            # Log order normalization and acceptance
            for pwr in issued_orders:
                logger.info(f"\n[DEBUG] Order processing for {pwr}:")
                if pwr in order_maps:
                    for raw_order, normalized in order_maps[pwr].items():
                        status = "accepted" if normalized in engine_accepted_orders.get(pwr, []) else "rejected"
                        logger.info(f"  {raw_order} -> {normalized} ({status})")
                
            logger.info(f"\n[DEBUG] Full issued orders: {issued_orders}")
            logger.info(f"[DEBUG] Full accepted orders: {engine_accepted_orders}")


        # Log the results
        #scores = env.compute_score()
        logger.info(f"Scores after {current_phase}: {scores}")
        
        if env.is_game_done():
            logger.info("Game finished due to normal conclusion.")
            break

        asyncio.sleep(1)

    #final_scores = env.compute_score()
    #logger.info(f"Final Scores: {final_scores}")
    #save_game_state(args.game_id, env, agents)
    logger.info("Done.")

def run():
    asyncio.run(main())

if __name__ == "__main__":
    run()
