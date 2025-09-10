from datetime import datetime, time
import json
from typing import Dict, List, Optional
import time as time_module

class EnergyManager:
    def __init__(self, config_file: str = 'energy_config.json'):
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Initialize zone states and timers
        self.zone_states: Dict[int, bool] = {}  # True if occupied
        self.zone_timers: Dict[int, Optional[float]] = {}
        self.active_zones = 0
        self.total_zones = 0
        
        # Initialize zones
        self._setup_zones()

    def _load_config(self, config_file: str) -> dict:
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Default configuration
            config = {
                'zones': {
                    '1': {'pin': 17, 'timeout': 300},  # 5 minutes
                    '2': {'pin': 27, 'timeout': 300},
                    '3': {'pin': 22, 'timeout': 300}
                },
                'schedule': {
                    'working_hours': {
                        'start': '08:00',
                        'end': '18:00'
                    }
                }
            }
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=4)
            return config

    def _setup_zones(self):
        """Initialize zone states without GPIO"""
        for zone_id, zone_config in self.config['zones'].items():
            zone_id = int(zone_id)
            self.zone_states[zone_id] = False
            self.zone_timers[zone_id] = None
            self.total_zones += 1

    def update_zone_status(self, zone_id: int, is_occupied: bool):
        """Update the occupancy status of a zone"""
        if zone_id in self.zone_states:
            was_occupied = self.zone_states[zone_id]
            self.zone_states[zone_id] = is_occupied
            
            if is_occupied and not was_occupied:
                self.active_zones += 1
                self._activate_zone(zone_id)
            elif not is_occupied and was_occupied:
                self.active_zones -= 1
                self._deactivate_zone(zone_id)

    def _activate_zone(self, zone_id: int):
        """Mock activation of devices in a zone"""
        pass

    def _deactivate_zone(self, zone_id: int):
        """Mock deactivation of devices in a zone"""
        pass

    def _is_within_working_hours(self) -> bool:
        """Check if current time is within working hours"""
        current_time = datetime.now().time()
        schedule = self.config['schedule']['working_hours']
        start_time = datetime.strptime(schedule['start'], '%H:%M').time()
        end_time = datetime.strptime(schedule['end'], '%H:%M').time()
        
        return start_time <= current_time <= end_time

    def _monitor_zones(self):
        """Monitor zones for timeout and schedule"""
        while self.running:
            current_time = time_module.time()
            
            for zone_id, last_empty_time in self.zone_timers.items():
                if not self.zone_states[zone_id] and last_empty_time is not None:
                    zone_config = self.config['zones'][str(zone_id)]
                    timeout = zone_config['timeout']
                    
                    if current_time - last_empty_time >= timeout:
                        self._deactivate_zone(zone_id)
                        self.zone_timers[zone_id] = None
            
            time_module.sleep(1)  # Check every second

    def cleanup(self):
        """Cleanup resources"""
        self.zone_states.clear()
        self.zone_timers.clear()

    def get_zone_status(self) -> Dict[int, bool]:
        """Get the current status of all zones"""
        return self.zone_states.copy()

    def get_energy_stats(self) -> Dict[str, any]:
        """Get energy usage statistics"""
        stats = {
            'active_zones': sum(1 for state in self.zone_states.values() if state),
            'total_zones': len(self.zone_states),
            'working_hours': self.config['schedule']['working_hours'],
            'occupancy_rate': sum(1 for state in self.zone_states.values() if state) / len(self.zone_states) if len(self.zone_states) > 0 else 0
        }
        return stats
