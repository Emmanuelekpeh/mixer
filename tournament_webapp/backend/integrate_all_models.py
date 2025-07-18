#!/usr/bin/env python3
print('Script started successfully')
"""
Integrate all discovered models into the tournament system
"""

import json
from .model_integration_system import ModelIntegrationSystem

def main():
    print('Starting model integration...')
    integrator = ModelIntegrationSystem()
    try:
        result = integrator.discover_and_integrate_all()
        with open('integration_result.json', 'w') as f:
            json.dump(result, f, indent=2)
    except Exception as e:
        print(f'Error during integration: {str(e)}')
    print('Integration complete!')

main() 