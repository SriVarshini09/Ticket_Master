"""
Master Ticket Router with Three Intelligence Tiers:
Tier 1: Rule-based (always works)
Tier 2: Lightweight ML (works with basic Python)
Tier 3: OpenAI (requires API key)
"""

from flask import Flask, request, jsonify, render_template
from datetime import datetime
import json
import os
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import logging

# Try to import AI components
try:
    from simple_ml_classifier import SimplifiedTicketAgent
    LIGHTWEIGHT_ML_AVAILABLE = True
    print("🧠 Lightweight ML available!")
except ImportError as e:
    LIGHTWEIGHT_ML_AVAILABLE = False
    print(f"📊 Lightweight ML not available: {e}")

try:
    import openai
    OPENAI_AVAILABLE = True
    print("🤖 OpenAI available!")
except ImportError:
    OPENAI_AVAILABLE = False
    print("🔑 OpenAI not available (install openai package)")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@dataclass
class Ticket:
    id: str
    subject: str
    message: str
    customer_email: str
    priority: str
    category: str
    assigned_team: str
    assigned_agent: Optional[str]
    status: str
    created_at: str
    sentiment: str
    estimated_resolution_time: str
    confidence: float = 0.0
    ai_reasoning: str = ""
    suggested_actions: List[str] = None
    needs_escalation: bool = False
    intelligence_tier: str = "rule-based"

class OpenAIClassifier:
    """OpenAI-based classification with markdown fix"""
    
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
    
    def classify(self, subject: str, message: str) -> Dict:
        """Classify using OpenAI with markdown handling"""
        prompt = f"""
        Analyze this support ticket and respond with JSON:
        
        Subject: {subject}
        Message: {message}
        
        Respond in this exact format:
        {{
            "priority": "high|medium|low",
            "category": "billing|technical|account|general", 
            "sentiment": "positive|neutral|negative|frustrated",
            "confidence": 85,
            "reasoning": "Brief explanation"
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            
            # Get raw response
            raw_content = response.choices[0].message.content.strip()
            print(f"🔍 Raw OpenAI response: {raw_content}")
            
            # STRIP MARKDOWN - This is the fix!
            cleaned_content = self.strip_markdown(raw_content)
            print(f"✅ Cleaned content: {cleaned_content}")
            
            # Parse JSON
            result = json.loads(cleaned_content)
            result['model_type'] = 'openai_gpt'
            return result
        
        except Exception as e:
            print(f"🚨 OpenAI classification failed: {e}")
            raise

    def strip_markdown(self, content: str) -> str:
        """Remove markdown formatting from OpenAI response"""
        # Remove ```json and ``` markers
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*$', '', content)
        content = content.strip()
        return content

class MasterTicketRouter:
    """Master router that tries all three intelligence tiers"""
    
    def __init__(self):
        self.intelligence_tier = "rule-based"
        self.openai_classifier = None
        self.ml_agent = None
        
        # Initialize available AI systems
        if OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY'):
            try:
                self.openai_classifier = OpenAIClassifier(os.getenv('OPENAI_API_KEY'))
                self.intelligence_tier = "openai"
                logger.info("🤖 Using OpenAI (Tier 3)")
            except Exception as e:
                logger.warning(f"OpenAI failed: {e}")
        
        if LIGHTWEIGHT_ML_AVAILABLE and not self.openai_classifier:
            try:
                self.ml_agent = SimplifiedTicketAgent()
                self.intelligence_tier = "lightweight_ml"
                logger.info("🧠 Using Lightweight ML (Tier 2)")
            except Exception as e:
                logger.warning(f"ML Agent failed: {e}")
        
        if self.intelligence_tier == "rule-based":
            logger.info("📋 Using Rule-based System (Tier 1)")
        
        # Rule-based fallback components
        self.priority_keywords = {
            'high': ['urgent', 'emergency', 'critical', 'down', 'broken', 'crashed', 'error', 'failed'],
            'medium': ['issue', 'problem', 'help', 'support', 'question', 'slow'],
            'low': ['feature', 'request', 'suggestion', 'enhancement', 'feedback']
        }
        
        self.category_keywords = {
            'billing': ['payment', 'invoice', 'bill', 'charge', 'refund', 'subscription', 'pricing'],
            'technical': ['bug', 'error', 'crash', 'login', 'password', 'access', 'integration', 'api'],
            'account': ['profile', 'settings', 'account', 'user', 'permissions', 'registration'],
            'general': ['question', 'info', 'information', 'help', 'support']
        }
        
        self.team_assignments = {
            'billing': 'Finance Team',
            'technical': 'Engineering Team',
            'account': 'Customer Success Team',
            'general': 'General Support Team'
        }
        
        self.agent_assignments = {
            'Finance Team': ['sarah.j@company.com', 'mike.c@company.com'],
            'Engineering Team': ['alex.r@company.com', 'priya.p@company.com'],
            'Customer Success Team': ['emma.w@company.com', 'david.k@company.com'],
            'General Support Team': ['lisa.b@company.com', 'john.s@company.com']
        }
        
        self.ticket_counter = 1000
    
    def route_ticket(self, subject: str, message: str, customer_email: str) -> Ticket:
        """Route ticket using the best available AI tier"""
        
        # Tier 3: Try OpenAI first
        if self.openai_classifier:
            return self._route_with_openai(subject, message, customer_email)
        
        # Tier 2: Try Lightweight ML
        elif self.ml_agent:
            return self._route_with_ml(subject, message, customer_email)
        
        # Tier 1: Fallback to rules
        else:
            return self._route_with_rules(subject, message, customer_email)
    
    def _route_with_openai(self, subject: str, message: str, customer_email: str) -> Ticket:
        """Route using OpenAI (Tier 3)"""
        try:
            classification = self.openai_classifier.classify(subject, message)
            
            ticket_id = f"TKT-{self.ticket_counter}"
            self.ticket_counter += 1
            
            assigned_team = self.team_assignments[classification['category']]
            
            return Ticket(
                id=ticket_id,
                subject=subject,
                message=message,
                customer_email=customer_email,
                priority=classification['priority'],
                category=classification['category'],
                assigned_team=assigned_team,
                assigned_agent=self._assign_agent(assigned_team),
                status='open',
                created_at=datetime.now().isoformat(),
                sentiment=classification['sentiment'],
                estimated_resolution_time=self._estimate_time(classification['priority'], classification['category']),
                confidence=classification['confidence'],
                ai_reasoning=f"OpenAI: {classification['reasoning']}",
                suggested_actions=self._generate_openai_actions(classification),
                needs_escalation=classification['priority'] == 'high' or classification['sentiment'] == 'frustrated',
                intelligence_tier="openai"
            )
            
        except Exception as e:
            logger.error(f"OpenAI routing failed: {e}")
            return self._route_with_rules(subject, message, customer_email)
    
    def _route_with_ml(self, subject: str, message: str, customer_email: str) -> Ticket:
        """Route using Lightweight ML (Tier 2)"""
        try:
            result = self.ml_agent.process_ticket(subject, message, customer_email)
            classification = result['classification']
            
            ticket_id = f"TKT-{self.ticket_counter}"
            self.ticket_counter += 1
            
            return Ticket(
                id=ticket_id,
                subject=subject,
                message=message,
                customer_email=customer_email,
                priority=classification['priority'],
                category=classification['category'],
                assigned_team=result['assigned_team'],
                assigned_agent=self._assign_agent(result['assigned_team']),
                status='open',
                created_at=datetime.now().isoformat(),
                sentiment=classification['sentiment'],
                estimated_resolution_time=result['estimated_resolution_time'],
                confidence=classification['confidence'],
                ai_reasoning=f"Lightweight ML: Statistical analysis with {classification['confidence']}% confidence",
                suggested_actions=result['suggested_actions'],
                needs_escalation=result['needs_escalation'],
                intelligence_tier="lightweight_ml"
            )
            
        except Exception as e:
            logger.error(f"ML routing failed: {e}")
            return self._route_with_rules(subject, message, customer_email)
    
    def _route_with_rules(self, subject: str, message: str, customer_email: str) -> Ticket:
        """Route using rules (Tier 1)"""
        combined_text = f"{subject} {message}"
        
        priority = self._analyze_priority(combined_text)
        category = self._analyze_category(combined_text)
        sentiment = self._analyze_sentiment(combined_text)
        assigned_team = self.team_assignments[category]
        
        ticket_id = f"TKT-{self.ticket_counter}"
        self.ticket_counter += 1
        
        return Ticket(
            id=ticket_id,
            subject=subject,
            message=message,
            customer_email=customer_email,
            priority=priority,
            category=category,
            assigned_team=assigned_team,
            assigned_agent=self._assign_agent(assigned_team),
            status='open',
            created_at=datetime.now().isoformat(),
            sentiment=sentiment,
            estimated_resolution_time=self._estimate_time(priority, category),
            confidence=75.0,
            ai_reasoning="Rule-based keyword analysis",
            suggested_actions=self._generate_rule_actions(priority, category),
            needs_escalation=priority == 'high',
            intelligence_tier="rule-based"
        )
    
    def _analyze_priority(self, text: str) -> str:
        """Rule-based priority analysis"""
        text_lower = text.lower()
        
        high_score = sum(1 for keyword in self.priority_keywords['high'] if keyword in text_lower)
        medium_score = sum(1 for keyword in self.priority_keywords['medium'] if keyword in text_lower)
        
        if any(word in text_lower for word in ['urgent', 'emergency', 'critical']) or high_score >= 2:
            return 'high'
        elif medium_score > 0:
            return 'medium'
        else:
            return 'low'
    
    def _analyze_category(self, text: str) -> str:
        """Rule-based category analysis"""
        text_lower = text.lower()
        category_scores = {}
        
        for category, keywords in self.category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            category_scores[category] = score
        
        if max(category_scores.values()) > 0:
            return max(category_scores, key=category_scores.get)
        return 'general'
    
    def _analyze_sentiment(self, text: str) -> str:
        """Rule-based sentiment analysis"""
        text_lower = text.lower()
        
        negative_words = ['angry', 'frustrated', 'terrible', 'awful', 'hate', 'disappointed']
        positive_words = ['great', 'excellent', 'love', 'amazing', 'wonderful']
        
        negative_score = sum(1 for word in negative_words if word in text_lower)
        positive_score = sum(1 for word in positive_words if word in text_lower)
        
        if negative_score > 0:
            return 'frustrated' if negative_score > 1 else 'negative'
        elif positive_score > 0:
            return 'positive'
        else:
            return 'neutral'
    
    def _estimate_time(self, priority: str, category: str) -> str:
        """Estimate resolution time"""
        times = {
            'billing': {'high': '2 hours', 'medium': '4 hours', 'low': '1 day'},
            'technical': {'high': '4 hours', 'medium': '8 hours', 'low': '2 days'},
            'account': {'high': '1 hour', 'medium': '2 hours', 'low': '4 hours'},
            'general': {'high': '2 hours', 'medium': '6 hours', 'low': '1 day'}
        }
        return times.get(category, {}).get(priority, '1 day')
    
    def _assign_agent(self, team: str) -> str:
        """Assign agent using round-robin"""
        agents = self.agent_assignments.get(team, ['support@company.com'])
        return agents[self.ticket_counter % len(agents)]
    
    def _generate_openai_actions(self, classification: Dict) -> List[str]:
        """Generate actions for OpenAI classification"""
        actions = ["🤖 AI-powered routing"]
        
        if classification['priority'] == 'high':
            actions.append("⚡ Immediate escalation")
        
        if classification['sentiment'] in ['frustrated', 'negative']:
            actions.append("💬 Use empathetic response")
        
        actions.append("📧 Send AI-generated acknowledgment")
        return actions
    
    def _generate_rule_actions(self, priority: str, category: str) -> List[str]:
        """Generate actions for rule-based classification"""
        actions = ["📋 Rule-based routing"]
        
        if priority == 'high':
            actions.append("🚨 High priority alert")
        
        actions.append(f"👥 Route to {self.team_assignments[category]}")
        actions.append("📧 Send standard acknowledgment")
        return actions
    
    def get_system_status(self) -> Dict:
        """Get current system capabilities"""
        return {
            'active_tier': self.intelligence_tier,
            'available_tiers': {
                'rule_based': True,
                'lightweight_ml': LIGHTWEIGHT_ML_AVAILABLE,
                'openai': OPENAI_AVAILABLE and bool(os.getenv('OPENAI_API_KEY'))
            },
            'total_tickets_processed': self.ticket_counter - 1000,
            'capabilities': {
                'intelligent_classification': self.intelligence_tier != 'rule-based',
                'confidence_scoring': True,
                'sentiment_analysis': True,
                'escalation_detection': True,
                'ai_reasoning': True
            }
        }

# Initialize the master router
router = MasterTicketRouter()
tickets_db = []

@app.route('/')
def index():
    """Render the enhanced dashboard"""
    return render_template('index.html')

@app.route('/api/system/status', methods=['GET'])
def get_system_status():
    """Get system capabilities and AI tier status"""
    return jsonify(router.get_system_status())

@app.route('/api/tickets', methods=['POST'])
def create_ticket():
    """Create a new support ticket with multi-tier AI processing"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['subject', 'message', 'customer_email']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Route the ticket using best available AI
        ticket = router.route_ticket(
            subject=data['subject'],
            message=data['message'],
            customer_email=data['customer_email']
        )
        
        # Store ticket
        tickets_db.append(asdict(ticket))
        
        # Enhanced response with AI insights
        response = {
            'success': True,
            'ticket': asdict(ticket),
            'message': f'Ticket {ticket.id} created and routed to {ticket.assigned_team}',
            'ai_insights': {
                'intelligence_tier': ticket.intelligence_tier,
                'confidence': ticket.confidence,
                'reasoning': ticket.ai_reasoning,
                'escalation_needed': ticket.needs_escalation,
                'suggested_actions': ticket.suggested_actions or []
            }
        }
        
        return jsonify(response), 201
        
    except Exception as e:
        logger.error(f"Error creating ticket: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/tickets', methods=['GET'])
def get_tickets():
    """Get all tickets with optional filtering"""
    try:
        priority = request.args.get('priority')
        category = request.args.get('category')
        status = request.args.get('status')
        tier = request.args.get('tier')  # Filter by AI tier
        
        filtered_tickets = tickets_db.copy()
        
        if priority:
            filtered_tickets = [t for t in filtered_tickets if t['priority'] == priority]
        if category:
            filtered_tickets = [t for t in filtered_tickets if t['category'] == category]
        if status:
            filtered_tickets = [t for t in filtered_tickets if t['status'] == status]
        if tier:
            filtered_tickets = [t for t in filtered_tickets if t.get('intelligence_tier') == tier]
        
        return jsonify({
            'tickets': filtered_tickets,
            'total': len(filtered_tickets)
        })
        
    except Exception as e:
        logger.error(f"Error fetching tickets: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Get enhanced analytics with multi-tier AI insights"""
    try:
        if not tickets_db:
            return jsonify({
                'total_tickets': 0,
                'by_priority': {},
                'by_category': {},
                'by_status': {},
                'by_sentiment': {},
                'by_intelligence_tier': {},
                'ai_metrics': {
                    'average_confidence': 0,
                    'escalation_rate': 0,
                    'active_tier': router.intelligence_tier
                }
            })
        
        # Calculate comprehensive metrics
        total_tickets = len(tickets_db)
        
        # Standard breakdowns
        by_priority = {}
        by_category = {}
        by_status = {}
        by_sentiment = {}
        by_tier = {}
        
        confidence_scores = []
        escalation_count = 0
        
        for ticket in tickets_db:
            # Basic breakdowns
            by_priority[ticket['priority']] = by_priority.get(ticket['priority'], 0) + 1
            by_category[ticket['category']] = by_category.get(ticket['category'], 0) + 1
            by_status[ticket['status']] = by_status.get(ticket['status'], 0) + 1
            by_sentiment[ticket['sentiment']] = by_sentiment.get(ticket['sentiment'], 0) + 1
            
            # AI tier breakdown
            tier = ticket.get('intelligence_tier', 'unknown')
            by_tier[tier] = by_tier.get(tier, 0) + 1
            
            # AI metrics
            confidence_scores.append(ticket.get('confidence', 0))
            if ticket.get('needs_escalation', False):
                escalation_count += 1
        
        return jsonify({
            'total_tickets': total_tickets,
            'by_priority': by_priority,
            'by_category': by_category,
            'by_status': by_status,
            'by_sentiment': by_sentiment,
            'by_intelligence_tier': by_tier,
            'ai_metrics': {
                'average_confidence': round(sum(confidence_scores) / len(confidence_scores), 1),
                'escalation_rate': round((escalation_count / total_tickets) * 100, 1),
                'active_tier': router.intelligence_tier,
                'tier_performance': {
                    tier: {
                        'count': count,
                        'percentage': round((count / total_tickets) * 100, 1)
                    }
                    for tier, count in by_tier.items()
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Error generating analytics: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/system/upgrade', methods=['POST'])
def upgrade_system():
    """Attempt to upgrade to higher AI tier"""
    try:
        target_tier = request.json.get('tier')
        
        if target_tier == 'openai':
            api_key = request.json.get('api_key')
            if not api_key:
                return jsonify({'error': 'OpenAI API key required'}), 400
            
            # Try to initialize OpenAI
            try:
                test_classifier = OpenAIClassifier(api_key)
                router.openai_classifier = test_classifier
                router.intelligence_tier = 'openai'
                os.environ['OPENAI_API_KEY'] = api_key
                
                return jsonify({
                    'success': True,
                    'message': 'Upgraded to OpenAI (Tier 3)',
                    'new_tier': 'openai'
                })
            except Exception as e:
                return jsonify({'error': f'OpenAI upgrade failed: {str(e)}'}), 400
        
        elif target_tier == 'lightweight_ml':
            if not LIGHTWEIGHT_ML_AVAILABLE:
                return jsonify({'error': 'Lightweight ML not available'}), 400
            
            try:
                router.ml_agent = SimplifiedTicketAgent()
                router.intelligence_tier = 'lightweight_ml'
                router.openai_classifier = None  # Downgrade from OpenAI
                
                return jsonify({
                    'success': True,
                    'message': 'Switched to Lightweight ML (Tier 2)',
                    'new_tier': 'lightweight_ml'
                })
            except Exception as e:
                return jsonify({'error': f'ML upgrade failed: {str(e)}'}), 400
        
        elif target_tier == 'rule_based':
            router.intelligence_tier = 'rule_based'
            router.openai_classifier = None
            router.ml_agent = None
            
            return jsonify({
                'success': True,
                'message': 'Switched to Rule-based (Tier 1)',
                'new_tier': 'rule_based'
            })
        
        else:
            return jsonify({'error': 'Invalid tier specified'}), 400
            
    except Exception as e:
        logger.error(f"Error upgrading system: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check with multi-tier AI status"""
    status = router.get_system_status()
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'intelligence_tier': status['active_tier'],
        'available_tiers': status['available_tiers'],
        'capabilities': status['capabilities'],
        'tickets_processed': status['total_tickets_processed']
    })

if __name__ == '__main__':
    # Create templates directory
    os.makedirs('templates', exist_ok=True)
    
    # Show comprehensive system status
    status = router.get_system_status()
    print(f"\n🎯 MULTI-TIER AI SYSTEM STATUS")
    print(f"{'='*50}")
    print(f"Active Tier: {status['active_tier'].upper()}")
    print(f"Available Tiers:")
    for tier, available in status['available_tiers'].items():
        emoji = "✅" if available else "❌"
        print(f"   {emoji} {tier.replace('_', ' ').title()}")
    
    print(f"\nCapabilities:")
    for capability, enabled in status['capabilities'].items():
        emoji = "✅" if enabled else "❌"
        print(f"   {emoji} {capability.replace('_', ' ').title()}")
    
    # Create diverse sample tickets to showcase all tiers
    sample_tickets = [
        {
            'subject': 'CRITICAL: Payment gateway completely down!',
            'message': 'Our entire payment system crashed during peak hours! Customers cannot complete purchases and we are losing massive revenue every second. This is an absolute emergency!',
            'customer_email': 'ceo@ecommerce.com'
        },
        {
            'subject': 'Cannot reset my password',
            'message': 'I tried the password reset link multiple times but never receive the email. Can someone help me regain access to my account?',
            'customer_email': 'user@customer.com'
        },
        {
            'subject': 'Loving the new dashboard design!',
            'message': 'The recent updates look fantastic! Would it be possible to add a dark mode option for night-time use?',
            'customer_email': 'happy.user@gmail.com'
        },
        {
            'subject': 'Double billing issue this month',
            'message': 'I was charged twice for my subscription on the 1st and 15th. Please investigate and process a refund for the duplicate charge.',
            'customer_email': 'billing@business.com'
        }
    ]
    
    # Create sample tickets showcasing AI tiers
    print(f"\n📋 CREATING SAMPLE TICKETS...")
    for i, sample in enumerate(sample_tickets, 1):
        ticket = router.route_ticket(
            subject=sample['subject'],
            message=sample['message'],
            customer_email=sample['customer_email']
        )
        tickets_db.append(asdict(ticket))
        
        print(f"   {i}. {ticket.id}: {ticket.priority.upper()}/{ticket.category}")
        print(f"      Tier: {ticket.intelligence_tier} (Confidence: {ticket.confidence}%)")
        print(f"      Team: {ticket.assigned_team}")
        print(f"      Reasoning: {ticket.ai_reasoning}")
    
    print(f"\n🚀 SYSTEM READY!")
    print(f"   Dashboard: http://127.0.0.1:5000")
    print(f"   System Status: http://127.0.0.1:5000/api/system/status")
    print(f"   Health Check: http://127.0.0.1:5000/health")
    
    app.run(debug=True, host='0.0.0.0', port=5000)