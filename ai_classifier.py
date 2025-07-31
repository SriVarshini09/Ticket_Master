import openai
import json
from typing import Dict, Optional
import os
from dataclasses import dataclass

@dataclass
class TicketClassification:
    priority: str
    category: str
    urgency_score: float
    sentiment: str
    suggested_actions: list
    confidence: float
    reasoning: str

class AITicketClassifier:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the AI classifier with OpenAI API"""
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv('OPENAI_API_KEY')
        )
        
        # Define the classification prompt
        self.system_prompt = """
        You are an expert customer support AI agent that analyzes support tickets and makes intelligent routing decisions.
        
        Your task is to analyze incoming support tickets and provide:
        1. Priority level (high, medium, low)
        2. Category (billing, technical, account, general)
        3. Urgency score (0-100)
        4. Customer sentiment (positive, neutral, negative, frustrated)
        5. Suggested immediate actions
        6. Confidence level (0-100)
        7. Brief reasoning for your decisions
        
        Consider these factors:
        - Business impact and urgency indicators
        - Technical complexity and severity
        - Customer emotional state
        - Historical patterns and context
        - SLA requirements
        
        Respond in JSON format only.
        """
    
    def classify_ticket(self, subject: str, message: str, customer_email: str = "") -> TicketClassification:
        """Use AI to classify and analyze the support ticket"""
        
        # Prepare the user prompt
        user_prompt = f"""
        Analyze this support ticket:
        
        Subject: {subject}
        Message: {message}
        Customer: {customer_email}
        
        Provide classification in this exact JSON format:
        {{
            "priority": "high|medium|low",
            "category": "billing|technical|account|general",
            "urgency_score": 0-100,
            "sentiment": "positive|neutral|negative|frustrated",
            "suggested_actions": ["action1", "action2", "action3"],
            "confidence": 0-100,
            "reasoning": "Brief explanation of your analysis"
        }}
        """
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # More cost-effective than GPT-4
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent results
                max_tokens=500
            )
            
            # Parse the response
            result = json.loads(response.choices[0].message.content)
            
            return TicketClassification(
                priority=result['priority'],
                category=result['category'],
                urgency_score=result['urgency_score'],
                sentiment=result['sentiment'],
                suggested_actions=result['suggested_actions'],
                confidence=result['confidence'],
                reasoning=result['reasoning']
            )
            
        except Exception as e:
            # Fallback to rule-based classification if AI fails
            print(f"AI classification failed: {e}")
            return self._fallback_classification(subject, message)
    
    def _fallback_classification(self, subject: str, message: str) -> TicketClassification:
        """Fallback rule-based classification if AI is unavailable"""
        text = f"{subject} {message}".lower()
        
        # Simple rule-based fallback
        if any(word in text for word in ['urgent', 'critical', 'emergency', 'down']):
            priority = 'high'
            urgency_score = 85
        elif any(word in text for word in ['issue', 'problem', 'help']):
            priority = 'medium'
            urgency_score = 60
        else:
            priority = 'low'
            urgency_score = 30
        
        if any(word in text for word in ['payment', 'billing', 'charge', 'refund']):
            category = 'billing'
        elif any(word in text for word in ['login', 'password', 'access', 'bug']):
            category = 'technical'
        elif any(word in text for word in ['account', 'profile', 'settings']):
            category = 'account'
        else:
            category = 'general'
        
        return TicketClassification(
            priority=priority,
            category=category,
            urgency_score=urgency_score,
            sentiment='neutral',
            suggested_actions=['Route to appropriate team', 'Send acknowledgment'],
            confidence=70,
            reasoning='Fallback rule-based classification used'
        )

# Enhanced AI Agent with Learning Capabilities
class SmartTicketAgent:
    def __init__(self, api_key: Optional[str] = None):
        self.classifier = AITicketClassifier(api_key)
        self.learning_data = []  # Store classifications for learning
        
    def process_ticket(self, subject: str, message: str, customer_email: str) -> Dict:
        """Process ticket with AI analysis and learning"""
        
        # Get AI classification
        classification = self.classifier.classify_ticket(subject, message, customer_email)
        
        # Determine team assignment based on AI category
        team_mapping = {
            'billing': 'Finance Team',
            'technical': 'Engineering Team', 
            'account': 'Customer Success Team',
            'general': 'General Support Team'
        }
        
        assigned_team = team_mapping.get(classification.category, 'General Support Team')
        
        # Generate intelligent response template
        response_template = self._generate_response_template(classification)
        
        # Store for learning (in production, save to database)
        self.learning_data.append({
            'input': {'subject': subject, 'message': message, 'email': customer_email},
            'classification': classification,
            'timestamp': '2025-01-30T00:00:00Z'
        })
        
        return {
            'classification': classification,
            'assigned_team': assigned_team,
            'response_template': response_template,
            'escalation_needed': classification.urgency_score > 80,
            'estimated_resolution': self._estimate_resolution_time(classification)
        }
    
    def _generate_response_template(self, classification: TicketClassification) -> str:
        """Generate intelligent response template based on classification"""
        
        templates = {
            'high': f"Thank you for contacting us. We understand this is urgent and have escalated your ticket. Expected response: 30 minutes.",
            'medium': f"Thank you for reaching out. We've received your {classification.category} request and will respond within 2 hours.",
            'low': f"Thank you for your {classification.category} inquiry. We'll get back to you within 24 hours."
        }
        
        return templates.get(classification.priority, "Thank you for contacting support.")
    
    def _estimate_resolution_time(self, classification: TicketClassification) -> str:
        """Intelligent resolution time estimation"""
        
        base_times = {
            ('high', 'technical'): '2-4 hours',
            ('high', 'billing'): '1-2 hours', 
            ('medium', 'technical'): '4-8 hours',
            ('medium', 'billing'): '2-4 hours',
            ('low', 'general'): '24-48 hours'
        }
        
        key = (classification.priority, classification.category)
        return base_times.get(key, '24 hours')
    
    def get_learning_insights(self) -> Dict:
        """Analyze learning data for insights"""
        if not self.learning_data:
            return {'message': 'No learning data available yet'}
        
        # Analyze patterns (simplified)
        categories = [item['classification'].category for item in self.learning_data]
        priorities = [item['classification'].priority for item in self.learning_data]
        
        return {
            'total_tickets_processed': len(self.learning_data),
            'most_common_category': max(set(categories), key=categories.count),
            'most_common_priority': max(set(priorities), key=priorities.count),
            'average_confidence': sum(item['classification'].confidence for item in self.learning_data) / len(self.learning_data)
        }

# Example usage and integration
if __name__ == "__main__":
    # Initialize the AI agent
    agent = SmartTicketAgent()
    
    # Test with sample tickets
    test_tickets = [
        {
            'subject': 'URGENT: Payment gateway down',
            'message': 'Our entire payment system crashed during Black Friday! Customers cannot complete purchases. Revenue is being lost every minute!',
            'email': 'ceo@ecommerce.com'
        },
        {
            'subject': 'Password reset not working',
            'message': 'I clicked the password reset link but never received the email. Can you help?',
            'email': 'user@example.com'
        }
    ]
    
    for ticket in test_tickets:
        print(f"\n{'='*50}")
        print(f"Processing: {ticket['subject']}")
        print(f"{'='*50}")
        
        result = agent.process_ticket(
            ticket['subject'],
            ticket['message'], 
            ticket['email']
        )
        
        classification = result['classification']
        print(f"Priority: {classification.priority.upper()}")
        print(f"Category: {classification.category}")
        print(f"Sentiment: {classification.sentiment}")
        print(f"Urgency Score: {classification.urgency_score}/100")
        print(f"Confidence: {classification.confidence}%")
        print(f"Assigned Team: {result['assigned_team']}")
        print(f"Escalation Needed: {result['escalation_needed']}")
        print(f"Suggested Actions: {', '.join(classification.suggested_actions)}")
        print(f"AI Reasoning: {classification.reasoning}")
        print(f"Estimated Resolution: {result['estimated_resolution']}")
    
    # Show learning insights
    print(f"\n{'='*50}")
    print("LEARNING INSIGHTS")
    print(f"{'='*50}")
    insights = agent.get_learning_insights()
    for key, value in insights.items():
        print(f"{key.replace('_', ' ').title()}: {value}")