import numpy as np
import heapq

class FishTracker:
    def __init__(self, threshold=5.0, lambda1=0.1, lambda2=0.1):
        self.threshold = threshold
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.disappeared_fishes = []  # To store disappeared fish (use a deque for efficiency)
        self.max_disappeared_frames = 2

    def calculate_basic_loss(self, fish1, fish2):
        # Position Loss
        position_loss = np.linalg.norm(np.array(fish1['position']) - np.array(fish2['position']))
        
        # Size Loss
        size_loss = np.abs(fish1['size'] - fish2['size'])
        
        # L2 Regularization terms
        position_reg = self.lambda1 * (position_loss ** 2)
        size_reg = self.lambda2 * (size_loss ** 2)
        
        # Combined Loss
        total_loss = position_loss + size_loss + position_reg + size_reg
        
        return total_loss
    def calculate_speed_loss(self, fish1, fish2, time_elapsed):
        # Calculate expected position of fish2 based on its speed
        if np.linalg.norm(fish2['speed']) == 0:
            return self.calculate_basic_loss(fish1, fish2)
        
        expected_position = np.array(fish2['position']) + np.array(fish2['speed']) * time_elapsed
        position_loss = np.linalg.norm(np.array(fish1['position']) - expected_position)

        # Size Loss
        size_loss = np.abs(fish1['size'] - fish2['size'])

        # L2 Regularization terms
        position_reg = self.lambda1 * (position_loss ** 2)
        size_reg = self.lambda2 * (size_loss ** 2)
        
        # Combined Loss
        total_loss = position_loss + size_loss + position_reg + size_reg
        
        return total_loss

    def update_disappeared_fishes(self, current_frame, fishes):
        # Remove fishes that have been disappeared for more than max_disappeared_frames
        self.disappeared_fishes = [
            fish for fish in self.disappeared_fishes
            if current_frame - fish['frame'] <= self.max_disappeared_frames
        ]
        
        # Add new disappeared fishes
        for fish in fishes:
            if fish['disappeared']:
                self.disappeared_fishes.append({'fish': fish, 'frame': current_frame})

    def track_fish(self, current_frame, new_fishes, basic = False,  time_elapsed=1.0):
        potential_matches = []
        
        for new_fish in new_fishes:
            if new_fish['new']:
                for disappeared_fish in self.disappeared_fishes:
                    if basic:
                      loss = self.calculate_basic_loss(new_fish, disappeared_fish['fish'])
                    else:
                      loss = self.calculate_speed_loss(new_fish, disappeared_fish['fish'], time_elapsed)
                    if loss < self.threshold:
                        heapq.heappush(potential_matches, (loss, new_fish, disappeared_fish))
        
        used_disappeared_fishes = set()
        updated_disappeared_fishes = []

        while potential_matches:
            loss, new_fish, disappeared_fish = heapq.heappop(potential_matches)
            if disappeared_fish['fish']['id'] not in used_disappeared_fishes and new_fish['id'] is None:
                new_fish['id'] = disappeared_fish['fish']['id']
                new_fish['new'] = False
                used_disappeared_fishes.add(disappeared_fish['fish']['id'])
            else:
                updated_disappeared_fishes.append(disappeared_fish)

        self.disappeared_fishes = [fish for fish in self.disappeared_fishes if fish['fish']['id'] not in used_disappeared_fishes]
