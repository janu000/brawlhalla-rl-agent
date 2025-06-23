from itertools import combinations
from pynput import keyboard

from BrawlhallaController import BrawlhallaController

def get_best_valid_action(active_keys, valid_actions, action_idx_map):
    valid_set = set(valid_actions)
    matched_sets = []

     # Find all valid subsets
    matched_sets = [
        s for s in valid_actions
        if s.issubset(active_keys)
    ]

    if matched_sets:
        best_set = matched_sets[0]  # because _valid_actions is ordered
        return action_idx_map[best_set]
    
    return []

def _to_canonical_key(key):
        """Converts a pynput key object or string to a canonical form for comparison."""
        if isinstance(key, keyboard.Key):
            return key # e.g., Key.space, Key.shift_l
        elif isinstance(key, keyboard.KeyCode):
            # Convert KeyCode to its character or itself if no char (e.g., F1)
            return key.char.lower() if key.char else key
        elif isinstance(key, str):
            return key.lower() # Already a char string
        return key # Fallback for unexpected types


controller = BrawlhallaController()

valid_actions = []
action_idx_map = {}
for action_idx, mapped_keys_list in controller.ACTION_MAPPER.items():
    canonical_mapped_keys = frozenset(_to_canonical_key(k) for k in mapped_keys_list)
    valid_actions.append(canonical_mapped_keys)
    action_idx_map[canonical_mapped_keys] = action_idx


priority_order = ['j', 'w', 's', 'a', 'd']  # higher priority = earlier in list
# Priority map for quick lookup
priority_map = {k: i for i, k in enumerate(priority_order)}

active_keys = {'w', 'a', 's', 'd', 'j'}

best = get_best_valid_action(active_keys, valid_actions, action_idx_map)


print(best)  # Output: [{'a', 'd'}] if 'a' outranks 's', or [{'s'}] if no better combo

