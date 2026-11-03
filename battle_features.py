import pandas as pd
import numpy as np

def extract_battle_features(data):
    """
    PART 3: Battle Features Extraction Function
    This function loops through the battle timeline (first 30 turns) and 
    summarizes what happened into numerical features for our ML model.
    """
    feature_list = []
    
    for battle in data:
        # Create a dictionary to store features for this specific battle
        features = {}
        
        battle_id = battle.get('battle_id')
        timeline = battle.get('battle_timeline', [])
        
        # --- Initialize counters for our 5 requirements ---
        p1_faints = 0
        p2_faints = 0
        p1_statuses = 0
        p2_statuses = 0
        p1_switches = 0
        p2_switches = 0
        p1_attacks = 0
        p2_attacks = 0
        p1_status_moves = 0
        p2_status_moves = 0
        
        # Lists to track trends (early game vs late game HP)
        p1_hp_early = []
        p2_hp_early = []
        p1_hp_late = []
        p2_hp_late = []
        
        last_p1_name = None
        last_p2_name = None
        
        # --- Loop through each turn in the 30-turn timeline ---
        for idx, turn in enumerate(timeline):
            turn_num = turn.get('turn', idx + 1)
            
            p1_state = turn.get('p1_pokemon_state', {})
            p2_state = turn.get('p2_pokemon_state', {})
            p1_move = turn.get('p1_move_details')
            p2_move = turn.get('p2_move_details')
            
            # 1. TRACK SWITCHES: Check if the active Pokémon's name changed from the previous turn
            curr_p1_name = p1_state.get('name')
            if last_p1_name and curr_p1_name != last_p1_name:
                p1_switches += 1
            last_p1_name = curr_p1_name
            
            curr_p2_name = p2_state.get('name')
            if last_p2_name and curr_p2_name != last_p2_name:
                p2_switches += 1
            last_p2_name = curr_p2_name
            
            # 2. TRACK REMAINING HP & FAINTS: Check hp_pct
            p1_hp = p1_state.get('hp_pct', 1.0)
            p2_hp = p2_state.get('hp_pct', 1.0)
            
            if p1_hp == 0.0:
                p1_faints += 1
            if p2_hp == 0.0:
                p2_faints += 1
                
            # 3. TRACK STATUS EFFECTS: Check if status is NOT 'nostatus' (sleep, poison, burn, paralysis)
            if p1_state.get('status', 'nostatus') != 'nostatus':
                p1_statuses += 1
            if p2_state.get('status', 'nostatus') != 'nostatus':
                p2_statuses += 1
                
            # 4. TRACK MOVE CATEGORIES: Check if move is attacking vs status move
            if p1_move:
                if p1_move.get('category') == 'status':
                    p1_status_moves += 1
                else:
                    p1_attacks += 1
            if p2_move:
                if p2_move.get('category') == 'status':
                    p2_status_moves += 1
                else:
                    p2_attacks += 1
                    
            # 5. TRACK TRENDS (Comebacks): Split timeline into early (1-15) and late (16-30)
            if turn_num <= 15:
                p1_hp_early.append(p1_hp)
                p2_hp_early.append(p2_hp)
            else:
                p1_hp_late.append(p1_hp)
                p2_hp_late.append(p2_hp)
        
        # --- Finalize and store features for this battle ---
        features['battle_id'] = battle_id
        
        # Requirement 1: Remaining HP at turn 30 and difference[cite: 25]
        features['p1_final_hp'] = p1_hp
        features['p2_final_hp'] = p2_hp
        features['hp_diff'] = p1_hp - p2_hp
        
        # Requirement 2: Knockouts (faints)[cite: 25]
        features['p1_faints'] = p1_faints
        features['p2_faints'] = p2_faints
        features['faints_diff'] = p2_faints - p1_faints # Positive means P2 lost more (good for P1)
        
        # Requirement 3: Status effects given/received[cite: 25]
        features['p1_statuses'] = p1_statuses
        features['p2_statuses'] = p2_statuses
        
        # Requirement 4: Switches, attacks, and status moves[cite: 25]
        features['p1_switches'] = p1_switches
        features['p2_switches'] = p2_switches
        features['p1_attacks'] = p1_attacks
        features['p2_attacks'] = p2_attacks
        features['p1_status_moves'] = p1_status_moves
        features['p2_status_moves'] = p2_status_moves
        
        # Requirement 5: Trend (Momentum / Comeback analysis)[cite: 25]
        p1_early_avg = sum(p1_hp_early) / len(p1_hp_early) if p1_hp_early else 1.0
        p1_late_avg = sum(p1_hp_late) / len(p1_hp_late) if p1_hp_late else 1.0
        p2_early_avg = sum(p2_hp_early) / len(p2_hp_early) if p2_hp_early else 1.0
        p2_late_avg = sum(p2_hp_late) / len(p2_hp_late) if p2_hp_late else 1.0
        
        features['p1_hp_trend'] = p1_late_avg - p1_early_avg
        features['p2_hp_trend'] = p2_late_avg - p2_early_avg
        
        # Include target variable if present in data
        if 'player_won' in battle:
            features['player_won'] = int(battle['player_won'])
            
        feature_list.append(features)
        
    # Convert the list of dictionaries into a clean pandas DataFrame and fill missing values with 0
    return pd.DataFrame(feature_list).fillna(0)
