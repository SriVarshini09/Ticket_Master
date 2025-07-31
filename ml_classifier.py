import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import re
from typing import Dict, Tuple, List
import joblib

class MLTicketClassifier:
    """Machine Learning-based ticket classifier that learns from data"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        
        self.priority_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.category_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.sentiment_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        
        self.is_trained = False
        self.feature_names = []
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text data"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def create_training_data(self) -> pd.DataFrame:
        """Generate synthetic training data for demonstration"""
        # In production, this would come from your actual ticket database
        training_data = [
            # High priority technical
            ("URGENT: Server crashed", "Production server is down, all services offline", "high", "technical", "negative"),
            ("CRITICAL: Database error", "Database connection failed, users cannot login", "high", "technical", "frustrated"),
            ("Emergency: Payment gateway down", "Cannot process payments, losing revenue", "high", "billing", "frustrated"),
            ("Site completely broken", "Website won't load for any users", "high", "technical", "negative"),
            ("URGENT: Security breach", "Possible unauthorized access detected", "high", "technical", "negative"),
            
            # Medium priority 
            ("Login issues", "Cannot log into my account, password reset not working", "medium", "technical", "neutral"),
            ("Billing question", "Charged wrong amount on my credit card", "medium", "billing", "neutral"),
            ("Feature not working", "Export function is not responding", "medium", "technical", "neutral"),
            ("Account access problem", "Cannot update my profile information", "medium", "account", "neutral"),
            ("Integration help needed", "API documentation is unclear", "medium", "technical", "neutral"),
            
            # Low priority
            ("Feature request", "Would like dark mode option", "low", "general", "positive"),
            ("General question", "How do I change my password?", "low", "account", "neutral"),
            ("Documentation update", "User guide needs more examples", "low", "general", "neutral"),
            ("Enhancement suggestion", "Could you add export to Excel?", "low", "general", "positive"),
            ("Feedback", "Love the new dashboard design!", "low", "general", "positive"),
            
            # More billing examples
            ("Double charged", "Billed twice for the same subscription", "medium", "billing", "frustrated"),
            ("Refund request", "Need refund for cancelled service", "medium", "billing", "neutral"),
            ("Payment failed", "Credit card payment was declined", "medium", "billing", "neutral"),
            ("Invoice question", "Don't understand charges on invoice", "low", "billing", "neutral"),
            
            # More technical examples
            ("API returning 500 errors", "Getting server errors on user endpoint", "high", "technical", "frustrated"),
            ("Slow performance", "Application is very slow to load", "medium", "technical", "neutral"),
            ("Mobile app crashes", "App crashes when I try to upload photos", "medium", "technical", "frustrated"),
            ("Browser compatibility", "Site doesn't work in Safari", "low", "technical", "neutral"),
            
            # Account management
            ("Cannot delete account", "Delete button is not working", "medium", "account", "frustrated"),
            ("Profile sync issues", "Changes not saving to profile", "medium", "account", "neutral"),
            ("Permission problems", "Cannot access team settings", "medium", "account", "neutral"),
            ("Two-factor auth issues", "2FA codes not working", "high", "account", "frustrated"),
        ]
        
        df = pd.DataFrame(training_data, columns=['subject', 'message', 'priority', 'category', 'sentiment'])
        df['combined_text'] = df['subject'] + ' ' + df['message']
        df['combined_text'] = df['combined_text'].apply(self.preprocess_text)
        
        return df
    
    def train(self, df: pd.DataFrame = None):
        """Train the ML models"""
        if df is None:
            df = self.create_training_data()
        
        print("Training ML models...")
        
        # Prepare features
        X = self.vectorizer.fit_transform(df['combined_text'])
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # Prepare targets
        y_priority = df['priority']
        y_category = df['category'] 
        y_sentiment = df['sentiment']
        
        # Train classifiers
        self.priority_classifier.fit(X, y_priority)
        self.category_classifier.fit(X, y_category)
        self.sentiment_classifier.fit(X, y_sentiment)
        
        self.is_trained = True
        
        # Print training results
        priority_score = self.priority_classifier.score(X, y_priority)
        category_score = self.category_classifier.score(X, y_category)
        sentiment_score = self.sentiment_classifier.score(X, y_sentiment)
        
        print(f"✅ Training completed!")
        print(f"   Priority accuracy: {priority_score:.3f}")
        print(f"   Category accuracy: {category_score:.3f}")
        print(f"   Sentiment accuracy: {sentiment_score:.3f}")
    
    def predict(self, subject: str, message: str) -> Dict:
        """Predict ticket classification using trained models"""
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
        
        # Preprocess input
        combined_text = self.preprocess_text(f"{subject} {message}")
        X = self.vectorizer.transform([combined_text])
        
        # Make predictions
        priority_pred = self.priority_classifier.predict(X)[0]
        priority_proba = max(self.priority_classifier.predict_proba(X)[0])
        
        category_pred = self.category_classifier.predict(X)[0]
        category_proba = max(self.category_classifier.predict_proba(X)[0])
        
        sentiment_pred = self.sentiment_classifier.predict(X)[0]
        sentiment_proba = max(self.sentiment_classifier.predict_proba(X)[0])
        
        # Calculate confidence score
        confidence = (priority_proba + category_proba + sentiment_proba) / 3 * 100
        
        # Get feature importance for this prediction
        feature_importance = self._get_prediction_features(X)
        
        return {
            'priority': priority_pred,
            'category': category_pred,
            'sentiment': sentiment_pred,
            'confidence': round(confidence, 1),
            'priority_confidence': round(priority_proba * 100, 1),
            'category_confidence': round(category_proba * 100, 1),
            'sentiment_confidence': round(sentiment_proba * 100, 1),
            'key_features': feature_importance[:5],  # Top 5 features
            'model_type': 'machine_learning'
        }
    
    def _get_prediction_features(self, X) -> List[str]:
        """Get the most important features for this prediction"""
        # Get feature weights
        feature_weights = X.toarray()[0]
        
        # Get indices of non-zero features
        non_zero_indices = np.nonzero(feature_weights)[0]
        
        # Create feature importance pairs
        feature_importance = [(self.feature_names[i], feature_weights[i]) for i in non_zero_indices]
        
        # Sort by importance
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return [feature for feature, weight in feature_importance]
    
    def save_model(self, filepath: str):
        """Save trained models to disk"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained models")
        
        model_data = {
            'vectorizer': self.vectorizer,
            'priority_classifier': self.priority_classifier,
            'category_classifier': self.category_classifier,
            'sentiment_classifier': self.sentiment_classifier,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Models saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained models from disk"""
        model_data = joblib.load(filepath)
        
        self.vectorizer = model_data['vectorizer']
        self.priority_classifier = model_data['priority_classifier']
        self.category_classifier = model_data['category_classifier']
        self.sentiment_classifier = model_data['sentiment_classifier']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        
        print(f"✅ Models loaded from {filepath}")
    
    def retrain_with_feedback(self, subject: str, message: str, 
                            correct_priority: str, correct_category: str, correct_sentiment: str):
        """Update models with new feedback data (incremental learning)"""
        # In production, you'd accumulate feedback and retrain periodically
        print(f"📝 Feedback received: {subject} -> {correct_priority}/{correct_category}/{correct_sentiment}")
        print("   (In production, this would trigger model retraining)")

# Intelligent Agent that combines ML with business logic
class IntelligentTicketAgent:
    """Complete AI agent with ML classification and intelligent actions"""
    
    def __init__(self):
        self.classifier = MLTicketClassifier()
        self.agent_memory = []  # Store interactions for learning
        
        # Train the model on startup
        self.classifier.train()
        
        # Business rules for team assignment
        self.team_mapping = {
            'billing': 'Finance Team',
            'technical': 'Engineering Team',
            'account': 'Customer Success Team',
            'general': 'General Support Team'
        }
        
        # SLA rules based on priority
        self.sla_rules = {
            'high': {'response_time': '30 minutes', 'resolution_time': '4 hours'},
            'medium': {'response_time': '2 hours', 'resolution_time': '24 hours'},
            'low': {'response_time': '24 hours', 'resolution_time': '72 hours'}
        }
    
    def process_ticket(self, subject: str, message: str, customer_email: str) -> Dict:
        """Complete AI-powered ticket processing"""
        
        # Get ML classification
        classification = self.classifier.predict(subject, message)
        
        # Make intelligent decisions
        assigned_team = self.team_mapping[classification['category']]
        sla = self.sla_rules[classification['priority']]
        
        # Determine if escalation is needed
        needs_escalation = (
            classification['priority'] == 'high' or
            classification['sentiment'] in ['frustrated', 'negative'] or
            'urgent' in subject.lower() or 'emergency' in subject.lower()
        )
        
        # Generate intelligent actions
        suggested_actions = self._generate_actions(classification, needs_escalation)
        
        # Create response
        result = {
            'ticket_id': f"TKT-{len(self.agent_memory) + 1000}",
            'classification': classification,
            'assigned_team': assigned_team,
            'needs_escalation': needs_escalation,
            'sla': sla,
            'suggested_actions': suggested_actions,
            'auto_response': self._generate_auto_response(classification, customer_email),
            'next_steps': self._determine_next_steps(classification)
        }
        
        # Store in agent memory for learning
        self.agent_memory.append({
            'input': {'subject': subject, 'message': message, 'email': customer_email},
            'output': result,
            'timestamp': '2025-01-30T00:00:00Z'
        })
        
        return result
    
    def _generate_actions(self, classification: Dict, needs_escalation: bool) -> List[str]:
        """Generate context-aware action suggestions"""
        actions = []
        
        if needs_escalation:
            actions.append("🚨 Escalate to senior agent immediately")
        
        if classification['priority'] == 'high':
            actions.append("📞 Call customer within 30 minutes")
        
        if classification['sentiment'] in ['frustrated', 'negative']:
            actions.append("💬 Use empathetic language in response")
        
        if classification['category'] == 'billing':
            actions.append("💳 Review customer billing history")
        elif classification['category'] == 'technical':
            actions.append("🔧 Check system logs and error reports")
        elif classification['category'] == 'account':
            actions.append("👤 Verify customer identity and permissions")
        
        actions.append(f"📧 Send acknowledgment email")
        actions.append(f"⏰ Set follow-up reminder")
        
        return actions
    
    def _generate_auto_response(self, classification: Dict, customer_email: str) -> str:
        """Generate personalized auto-response"""
        priority = classification['priority']
        category = classification['category']
        sentiment = classification['sentiment']
        
        # Adjust tone based on sentiment
        if sentiment in ['frustrated', 'negative']:
            greeting = "We sincerely apologize for the inconvenience and understand your frustration."
        else:
            greeting = "Thank you for contacting our support team."
        
        # Priority-specific messaging
        if priority == 'high':
            urgency = "We've marked this as high priority and our team is working on it immediately."
        elif priority == 'medium':
            urgency = "We've received your request and will respond shortly."
        else:
            urgency = "We've received your inquiry and will get back to you soon."
        
        return f"{greeting} {urgency} You should expect a response within our standard timeframe for {category} issues."
    
    def _determine_next_steps(self, classification: Dict) -> List[str]:
        """Determine intelligent next steps"""
        steps = []
        
        if classification['confidence'] < 70:
            steps.append("🤔 Review classification - confidence is low")
        
        if classification['priority'] == 'high':
            steps.append("⚡ Begin immediate resolution process")
        
        steps.append(f"👥 Route to {self.team_mapping[classification['category']]}")
        steps.append("📋 Update ticket status to 'In Progress'")
        steps.append("📊 Log interaction in CRM system")
        
        return steps
    
    def get_agent_insights(self) -> Dict:
        """Analyze agent performance and learning"""
        if not self.agent_memory:
            return {'message': 'No tickets processed yet'}
        
        total_tickets = len(self.agent_memory)
        classifications = [item['output']['classification'] for item in self.agent_memory]
        
        priorities = [c['priority'] for c in classifications]
        categories = [c['category'] for c in classifications]
        confidence_scores = [c['confidence'] for c in classifications]
        
        return {
            'total_tickets_processed': total_tickets,
            'average_confidence': round(sum(confidence_scores) / len(confidence_scores), 1),
            'priority_distribution': {
                'high': priorities.count('high'),
                'medium': priorities.count('medium'), 
                'low': priorities.count('low')
            },
            'category_distribution': {
                'technical': categories.count('technical'),
                'billing': categories.count('billing'),
                'account': categories.count('account'),
                'general': categories.count('general')
            },
            'escalation_rate': sum(1 for item in self.agent_memory if item['output']['needs_escalation']) / total_tickets * 100
        }

# Demo the intelligent agent
if __name__ == "__main__":
    print("🤖 Initializing Intelligent Ticket Agent...")
    agent = IntelligentTicketAgent()
    
    # Test cases
    test_cases = [
        {
            'subject': 'CRITICAL: Payment system crashed',
            'message': 'Our entire payment processing is down! Customers cannot checkout. We are losing thousands in revenue every minute!',
            'email': 'ceo@ecommerce.com'
        },
        {
            'subject': 'Cannot login to account',
            'message': 'I forgot my password and the reset email never arrived. Can someone help me access my account?',
            'email': 'user@customer.com'
        },
        {
            'subject': 'Love the new features!',
            'message': 'The new dashboard looks amazing. Could you add a dark mode option in the future?',
            'email': 'happy.user@gmail.com'
        }
    ]
    
    print("\n" + "="*80)
    print("🎯 PROCESSING TICKETS WITH AI AGENT")
    print("="*80)
    
    for i, ticket in enumerate(test_cases, 1):
        print(f"\n📋 TICKET #{i}: {ticket['subject']}")
        print("-" * 60)
        
        result = agent.process_ticket(
            ticket['subject'],
            ticket['message'],
            ticket['email']
        )
        
        classification = result['classification']
        
        print(f"🎯 Classification:")
        print(f"   Priority: {classification['priority'].upper()} ({classification['priority_confidence']}% confidence)")
        print(f"   Category: {classification['category']}")
        print(f"   Sentiment: {classification['sentiment']}")
        print(f"   Overall Confidence: {classification['confidence']}%")
        
        print(f"\n🎪 Agent Decisions:")
        print(f"   Assigned Team: {result['assigned_team']}")
        print(f"   Escalation Needed: {'Yes' if result['needs_escalation'] else 'No'}")
        print(f"   Response Time SLA: {result['sla']['response_time']}")
        
        print(f"\n⚡ Suggested Actions:")
        for action in result['suggested_actions']:
            print(f"   • {action}")
        
        print(f"\n💬 Auto-Response Preview:")
        print(f"   {result['auto_response']}")
        
        if classification['key_features']:
            print(f"\n🔍 Key Decision Features: {', '.join(classification['key_features'][:3])}")
    
    # Show agent insights
    print("\n" + "="*80)
    print("📊 AGENT PERFORMANCE INSIGHTS")
    print("="*80)
    
    insights = agent.get_agent_insights()
    
    print(f"Total Tickets Processed: {insights['total_tickets_processed']}")
    print(f"Average ML Confidence: {insights['average_confidence']}%")
    print(f"Escalation Rate: {insights['escalation_rate']:.1f}%")
    
    print(f"\nPriority Distribution:")
    for priority, count in insights['priority_distribution'].items():
        print(f"   {priority.capitalize()}: {count}")
    
    print(f"\nCategory Distribution:")
    for category, count in insights['category_distribution'].items():
        print(f"   {category.capitalize()}: {count}")
    
    print("\n🎉 AI Agent successfully demonstrated intelligent ticket processing!")