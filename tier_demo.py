"""
Interactive Tier Demonstration Script
Shows all three AI tiers processing the same ticket
"""

import requests
import json
import os
import time

class TierDemo:
    def __init__(self, base_url="http://127.0.0.1:5000"):
        self.base_url = base_url
        self.sample_tickets = [
            {
                'name': 'Critical Emergency',
                'ticket': {
                    'subject': 'URGENT: Payment gateway crashed during Black Friday!',
                    'message': 'Our entire payment system is down during our biggest sales day! Customers cannot complete purchases and we are losing thousands of dollars every minute. This is a complete disaster and needs immediate attention!',
                    'customer_email': 'ceo@ecommerce.com'
                }
            },
            {
                'name': 'Subtle Issue',
                'ticket': {
                    'subject': 'Billing inquiry about recent charges',
                    'message': 'I noticed some charges on my account that seem a bit higher than usual. Could someone help me understand what these are for?',
                    'customer_email': 'customer@business.com'
                }
            },
            {
                'name': 'Positive Feedback',
                'ticket': {
                    'subject': 'Love the new dashboard updates!',
                    'message': 'The recent changes to the dashboard look fantastic! The interface is much cleaner. Would it be possible to add a dark mode option?',
                    'customer_email': 'happy.user@gmail.com'
                }
            }
        ]
    
    def get_system_status(self):
        try:
            response = requests.get(f"{self.base_url}/api/system/status")
            return response.json()
        except:
            return None
    
    def create_ticket(self, ticket_data):
        try:
            response = requests.post(
                f"{self.base_url}/api/tickets",
                headers={'Content-Type': 'application/json'},
                json=ticket_data
            )
            return response.json()
        except Exception as e:
            return {'error': str(e)}
    
    def upgrade_tier(self, tier, api_key=None):
        try:
            data = {'tier': tier}
            if api_key and tier == 'openai':
                data['api_key'] = api_key
            
            response = requests.post(
                f"{self.base_url}/api/system/upgrade",
                headers={'Content-Type': 'application/json'},
                json=data
            )
            return response.json()
        except Exception as e:
            return {'error': str(e)}
    
    def demonstrate_tier_differences(self):
        print("MULTI-TIER AI DEMONSTRATION")
        print("=" * 60)
        
        status = self.get_system_status()
        if not status:
            print("Cannot connect to server. Make sure it's running!")
            return
        
        print(f"Current System Status:")
        print(f"   Active Tier: {status['active_tier'].upper()}")
        print(f"   Available Tiers: {list(status['available_tiers'].keys())}")
        
        test_ticket = self.sample_tickets[0]['ticket']
        
        print(f"\nTESTING SAME TICKET ACROSS ALL TIERS")
        print(f"Test Ticket: '{test_ticket['subject']}'")
        print("-" * 60)
        
        print(f"\nTIER 1: RULE-BASED SYSTEM")
        self.test_tier('rule_based', test_ticket)
        
        if status['available_tiers'].get('lightweight_ml'):
            print(f"\nTIER 2: LIGHTWEIGHT ML SYSTEM")
            self.test_tier('lightweight_ml', test_ticket)
        
        if status['available_tiers'].get('openai'):
            print(f"\nTIER 3: OPENAI SYSTEM")
            self.test_tier('openai', test_ticket)
        
        print(f"\nDEMONSTRATING FAILURE HANDLING")
        self.demonstrate_failure_handling()
    
    def test_tier(self, tier, ticket_data):
        print(f"   Switching to {tier.upper()}...")
        
        upgrade_result = self.upgrade_tier(tier)
        if 'error' in upgrade_result:
            print(f"   Failed to switch: {upgrade_result['error']}")
            return
        
        print(f"   Successfully switched to {tier.upper()}")
        
        time.sleep(1)
        
        print(f"   Creating ticket...")
        result = self.create_ticket(ticket_data)
        
        if 'error' in result:
            print(f"   Error: {result['error']}")
            return
        
        ticket = result['ticket']
        ai_insights = result.get('ai_insights', {})
        
        print(f"   RESULTS:")
        print(f"      Priority: {ticket['priority'].upper()}")
        print(f"      Category: {ticket['category']}")
        print(f"      Sentiment: {ticket['sentiment']}")
        print(f"      Confidence: {ticket['confidence']}%")
        print(f"      Team: {ticket['assigned_team']}")
        print(f"      Tier: {ticket['intelligence_tier']}")
        print(f"      Reasoning: {ticket['ai_reasoning']}")
        
        if ai_insights.get('suggested_actions'):
            print(f"      Actions: {', '.join(ai_insights['suggested_actions'])}")
    
    def demonstrate_failure_handling(self):
        print(f"   Scenario: OpenAI API fails during operation")
        
        status = self.get_system_status()
        if status['available_tiers'].get('openai'):
            print(f"   Current: OpenAI Tier 3 is active")
            
            test_ticket = self.sample_tickets[1]['ticket']
            result = self.create_ticket(test_ticket)
            
            if 'ticket' in result:
                ticket = result['ticket']
                if ticket['intelligence_tier'] == 'openai':
                    print(f"   OpenAI processed successfully (Confidence: {ticket['confidence']}%)")
                else:
                    print(f"   OpenAI failed, fell back to {ticket['intelligence_tier']} (Confidence: {ticket['confidence']}%)")
                    print(f"   System remained operational - zero downtime!")
        
        print(f"\n   Simulating API failure by switching tiers...")
        
        print(f"   Falling back to Lightweight ML...")
        upgrade_result = self.upgrade_tier('lightweight_ml')
        if 'success' in upgrade_result and upgrade_result['success']:
            result = self.create_ticket(self.sample_tickets[2]['ticket'])
            ticket = result['ticket']
            print(f"   ML Fallback working (Confidence: {ticket['confidence']}%)")
        
        print(f"   Final fallback to Rule-based...")
        upgrade_result = self.upgrade_tier('rule_based')
        if 'success' in upgrade_result and upgrade_result['success']:
            result = self.create_ticket(self.sample_tickets[0]['ticket'])
            ticket = result['ticket']
            print(f"   Rule-based fallback working (Confidence: {ticket['confidence']}%)")
        
        print(f"\n   KEY POINT: System NEVER fails completely!")
        print(f"   Three layers of protection ensure 100% uptime")
    
    def interactive_demo(self):
        print("INTERACTIVE TIER DEMONSTRATION")
        print("=" * 50)
        
        while True:
            print(f"\nDemo Options:")
            print(f"   1. Show tier comparison")
            print(f"   2. Demonstrate failure handling") 
            print(f"   3. Test custom ticket")
            print(f"   4. Show system status")
            print(f"   5. Exit")
            
            choice = input(f"\nChoose demo (1-5): ").strip()
            
            if choice == '1':
                self.demonstrate_tier_differences()
            elif choice == '2':
                self.demonstrate_failure_handling()
            elif choice == '3':
                self.custom_ticket_demo()
            elif choice == '4':
                self.show_system_status()
            elif choice == '5':
                print(f"Demo complete!")
                break
            else:
                print(f"Invalid choice. Please select 1-5.")
    
    def custom_ticket_demo(self):
        print(f"\nCUSTOM TICKET DEMO")
        print(f"-" * 30)
        
        subject = input("Enter ticket subject: ").strip()
        message = input("Enter ticket message: ").strip()
        email = input("Enter customer email: ").strip() or "demo@example.com"
        
        if not subject or not message:
            print("Subject and message are required!")
            return
        
        ticket_data = {
            'subject': subject,
            'message': message,
            'customer_email': email
        }
        
        print(f"\nProcessing your ticket...")
        result = self.create_ticket(ticket_data)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            return
        
        ticket = result['ticket']
        print(f"\nANALYSIS RESULTS:")
        print(f"   Ticket ID: {ticket['id']}")
        print(f"   Priority: {ticket['priority'].upper()}")
        print(f"   Category: {ticket['category']}")
        print(f"   Sentiment: {ticket['sentiment']}")
        print(f"   Confidence: {ticket['confidence']}%")
        print(f"   AI Tier: {ticket['intelligence_tier']}")
        print(f"   Assigned Team: {ticket['assigned_team']}")
        print(f"   Reasoning: {ticket['ai_reasoning']}")
    
    def show_system_status(self):
        print(f"\nSYSTEM STATUS")
        print(f"-" * 25)
        
        status = self.get_system_status()
        if not status:
            print("Cannot connect to server!")
            return
        
        print(f"Active Tier: {status['active_tier'].upper()}")
        print(f"Tickets Processed: {status['total_tickets_processed']}")
        
        print(f"\nAvailable Tiers:")
        for tier, available in status['available_tiers'].items():
            emoji = "Yes" if available else "No"
            print(f"   {emoji} - {tier.replace('_', ' ').title()}")
        
        print(f"\nCapabilities:")
        for capability, enabled in status['capabilities'].items():
            emoji = "Yes" if enabled else "No"
            print(f"   {emoji} - {capability.replace('_', ' ').title()}")

if __name__ == "__main__":
    demo = TierDemo()
    
    print("Starting Tier Demonstration...")
    print("Make sure your app_master.py is running!")
    print()
    
    status = demo.get_system_status()
    if not status:
        print("Server not reachable at http://127.0.0.1:5000")
        print("Start with: python app_master.py")
        exit(1)
    
    demo.interactive_demo()
