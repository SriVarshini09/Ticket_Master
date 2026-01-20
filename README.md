#  Support Ticket Router

An intelligent customer support ticket routing and management system built with Flask. This application automatically analyzes incoming support tickets and routes them to the appropriate teams based on content, priority, and category.

##  Features

- **Intelligent Routing**: Automatically categorizes tickets (billing, technical, account, general)
- **Priority Detection**: Analyzes urgency and assigns priority levels (high, medium, low)
- **Sentiment Analysis**: Basic sentiment detection to flag negative customer experiences
- **Team Assignment**: Routes tickets to appropriate teams and agents
- **Real-time Dashboard**: Modern web interface for managing tickets
- **REST API**: Complete API for integration with other systems
- **Analytics**: Basic metrics and reporting capabilities

##  Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd support-ticket-router

# Run with Docker Compose
docker-compose up --build

# Access the application at http://localhost:5000
```

### Option 2: Local Development

```bash
# Clone the repository
git clone <your-repo-url>
cd support-ticket-router

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create templates directory
mkdir -p templates

# Run the application
python app.py

# Access the application at http://localhost:5000
```

##  Project Structure

```
support-ticket-router/
├── app.py                 # Main Flask application
├── templates/
│   └── index.html        # Frontend dashboard
├── requirements.txt      # Python dependencies
├── Dockerfile           # Container configuration
├── docker-compose.yml   # Local development setup
├── README.md           # This file
├── .gitignore          # Git ignore rules
└── tests/              # Test files (optional)
```

## 🔧 API Endpoints

### Create Ticket
```http
POST /api/tickets
Content-Type: application/json

{
    "subject": "Cannot access my account",
    "message": "I keep getting an error when trying to log in",
    "customer_email": "user@example.com"
}
```

### Get All Tickets
```http
GET /api/tickets?priority=high&category=technical&status=open
```

### Get Specific Ticket
```http
GET /api/tickets/TKT-1001
```

### Update Ticket Status
```http
PUT /api/tickets/TKT-1001/status
Content-Type: application/json

{
    "status": "resolved"
}
```

### Get Analytics
```http
GET /api/analytics
```

### Health Check
```http
GET /health
```

##  How It Works

### 1. Ticket Analysis
The system analyzes incoming tickets using keyword-based classification:

- **Priority Detection**: Scans for urgency indicators like "urgent", "emergency", "critical"
- **Category Classification**: Identifies topics like "billing", "technical", "account" issues
- **Sentiment Analysis**: Detects emotional tone to flag frustrated customers

### 2. Routing Logic
Based on the analysis:
- Routes to appropriate team (Finance, Engineering, Customer Success)
- Assigns to available agents using round-robin
- Sets estimated resolution time based on priority and category

### 3. Dashboard Features
- Create new tickets with automatic routing
- View and filter existing tickets
- Real-time analytics and metrics
- Responsive design for mobile and desktop

##  Deployment Options

### Heroku
1. Create a new Heroku app
2. Connect your GitHub repository
3. Deploy from the main branch
4. The app will automatically detect and use the Dockerfile

### Docker Hub + Cloud Platform
1. Build and push to Docker Hub:
```bash
docker build -t your-username/ticket-router .
docker push your-username/ticket-router
```

2. Deploy to your cloud platform using the Docker image

### Traditional VPS
1. Clone the repository on your server
2. Set up a reverse proxy (nginx)
3. Use gunicorn as the WSGI server
4. Set up systemd service for auto-restart

##  Testing

### Manual Testing
1. Open the dashboard at `http://localhost:5000`
2. Create test tickets with different content to see routing in action
3. Try various keywords to test priority and category detection

### API Testing with curl
```bash
# Create a high-priority technical ticket
curl -X POST http://localhost:5000/api/tickets \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "URGENT: Database connection failed",
    "message": "Our production database is down and customers cannot access their accounts. This is critical!",
    "customer_email": "admin@company.com"
  }'

# Get all high-priority tickets
curl "http://localhost:5000/api/tickets?priority=high"

# Get analytics
curl http://localhost:5000/api/analytics
```

## 🔮 Future Enhancements

### Short-term
- [ ] Add database persistence (PostgreSQL/MongoDB)
- [ ] Implement user authentication and authorization
- [ ] Add email notifications for ticket assignments
- [ ] Include file attachment support

### Medium-term
- [ ] Integrate with OpenAI API for better classification
- [ ] Add ticket escalation rules
- [ ] Implement SLA tracking and alerts
- [ ] Add customer self-service portal

### Long-term
- [ ] Machine learning models for improved routing
- [ ] Integration with popular ticketing systems (Zendesk, ServiceNow)
- [ ] Advanced analytics and reporting
- [ ] Multi-language support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📊 Performance Considerations

- **In-Memory Storage**: Current version uses in-memory storage for simplicity
- **Scalability**: For production, implement database persistence and caching
- **Rate Limiting**: Consider adding rate limiting for API endpoints
- **Monitoring**: Add logging and monitoring for production deployments

## 🛡️ Security Notes

- Input validation is implemented for all API endpoints
- Consider adding authentication for production use
- Implement CORS properly if serving from different domains
- Use HTTPS in production environments

## 📝 License

This project is open source and available under the MIT License.

## 📞 Support

For questions or issues:
1. Check the Issues section on GitHub
2. Review the API documentation above
3. Test with the provided examples

---


**Built for the modern support team** 🚀



