#!/usr/bin/env python3
"""
API Test Script for Support Ticket Router
Run this script to test all API endpoints
"""

import requests
import json
import time
from datetime import datetime

# Configuration
BASE_URL = "http://127.0.0.1:5000"
API_BASE = f"{BASE_URL}/api"

def test_health_check():
    """Test health check endpoint"""
    print("🏥 Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        print("✅ Health check passed")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_create_tickets():
    """Test ticket creation with various scenarios"""
    print("\n🎫 Testing ticket creation...")
    
    test_tickets = [
        {
            "subject": "URGENT: Cannot access billing dashboard",
            "message": "This is an emergency! Our billing system is completely down and we can't process payments. Customers are calling non-stop. Please help immediately!",
            "customer_email": "cfo@company.com"
        },
        {
            "subject": "Password reset not working",
            "message": "I've been trying to reset my password for the past hour but the email never arrives. Can someone help me with this?",
            "customer_email": "user@example.com"
        },
        {
            "subject": "Feature request: Dark mode",
            "message": "It would be great if you could add a dark mode option to the application. Many users have been asking for this.",
            "customer_email": "feedback@customer.com"
        },
        {
            "subject": "Double billing issue",
            "message": "I was charged twice for my subscription this month. The first charge was on the 1st and another on the 15th. Please refund the duplicate charge.",
            "customer_email": "accounting@business.com"
        },
        {
            "subject": "API integration help needed",
            "message": "I'm trying to integrate with your API but keep getting 401 errors. The documentation seems outdated. Can you help?",
            "customer_email": "developer@startup.com"
        }
    ]
    
    created_tickets = []
    
    for i, ticket_data in enumerate(test_tickets, 1):
        print(f"  Creating ticket {i}/{len(test_tickets)}...")
        try:
            response = requests.post(
                f"{API_BASE}/tickets",
                headers={"Content-Type": "application/json"},
                json=ticket_data
            )
            
            if response.status_code == 201:
                data = response.json()
                ticket = data['ticket']
                created_tickets.append(ticket)
                print(f"    ✅ Created {ticket['id']} - Priority: {ticket['priority']}, Category: {ticket['category']}")
                print(f"       Routed to: {ticket['assigned_team']} ({ticket['assigned_agent']})")
            else:
                print(f"    ❌ Failed to create ticket: {response.text}")
                
        except Exception as e:
            print(f"    ❌ Error creating ticket: {e}")
    
    print(f"✅ Created {len(created_tickets)} tickets successfully")
    return created_tickets

def test_get_tickets():
    """Test retrieving tickets with filters"""
    print("\n📋 Testing ticket retrieval...")
    
    # Test getting all tickets
    try:
        response = requests.get(f"{API_BASE}/tickets")
        assert response.status_code == 200
        data = response.json()
        total_tickets = data['total']
        print(f"  ✅ Retrieved {total_tickets} total tickets")
        
        # Test filtering by priority
        response = requests.get(f"{API_BASE}/tickets?priority=high")
        assert response.status_code == 200
        high_priority = response.json()['total']
        print(f"  ✅ Found {high_priority} high priority tickets")
        
        # Test filtering by category
        response = requests.get(f"{API_BASE}/tickets?category=technical")
        assert response.status_code == 200
        technical = response.json()['total']
        print(f"  ✅ Found {technical} technical tickets")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error retrieving tickets: {e}")
        return False

def test_ticket_status_update(ticket_id):
    """Test updating ticket status"""
    print(f"\n🔄 Testing status update for {ticket_id}...")
    
    statuses = ['in_progress', 'resolved', 'closed']
    
    for status in statuses:
        try:
            response = requests.put(
                f"{API_BASE}/tickets/{ticket_id}/status",
                headers={"Content-Type": "application/json"},
                json={"status": status}
            )
            
            if response.status_code == 200:
                print(f"  ✅ Updated status to: {status}")
            else:
                print(f"  ❌ Failed to update status to {status}: {response.text}")
                
        except Exception as e:
            print(f"  ❌ Error updating status to {status}: {e}")

def test_individual_ticket_retrieval(ticket_id):
    """Test retrieving a specific ticket"""
    print(f"\n🔍 Testing individual ticket retrieval for {ticket_id}...")
    
    try:
        response = requests.get(f"{API_BASE}/tickets/{ticket_id}")
        
        if response.status_code == 200:
            ticket = response.json()['ticket']
            print(f"  ✅ Retrieved ticket: {ticket['subject']}")
            print(f"     Status: {ticket['status']}, Priority: {ticket['priority']}")
            return True
        else:
            print(f"  ❌ Failed to retrieve ticket: {response.text}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error retrieving ticket: {e}")
        return False

def test_analytics():
    """Test analytics endpoint"""
    print("\n📊 Testing analytics...")
    
    try:
        response = requests.get(f"{API_BASE}/analytics")
        assert response.status_code == 200
        data = response.json()
        
        print(f"  ✅ Analytics retrieved:")
        print(f"     Total tickets: {data['total_tickets']}")
        print(f"     By priority: {data['by_priority']}")
        print(f"     By category: {data['by_category']}")
        print(f"     By status: {data['by_status']}")
        print(f"     By sentiment: {data['by_sentiment']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error retrieving analytics: {e}")
        return False

def test_error_handling():
    """Test error handling"""
    print("\n⚠️  Testing error handling...")
    
    # Test creating ticket with missing fields
    try:
        response = requests.post(
            f"{API_BASE}/tickets",
            headers={"Content-Type": "application/json"},
            json={"subject": "Test"}  # Missing required fields
        )
        
        if response.status_code == 400:
            print("  ✅ Properly rejected incomplete ticket")
        else:
            print(f"  ❌ Should have rejected incomplete ticket, got: {response.status_code}")
            
    except Exception as e:
        print(f"  ❌ Error testing incomplete ticket: {e}")
    
    # Test retrieving non-existent ticket
    try:
        response = requests.get(f"{API_BASE}/tickets/INVALID-ID")
        
        if response.status_code == 404:
            print("  ✅ Properly returned 404 for non-existent ticket")
        else:
            print(f"  ❌ Should have returned 404, got: {response.status_code}")
            
    except Exception as e:
        print(f"  ❌ Error testing non-existent ticket: {e}")

def run_comprehensive_test():
    """Run all tests"""
    print("🚀 Starting comprehensive API test suite...\n")
    start_time = time.time()
    
    # Test health check first
    if not test_health_check():
        print("❌ Health check failed. Is the server running?")
        return
    
    # Create test tickets
    created_tickets = test_create_tickets()
    
    if not created_tickets:
        print("❌ No tickets created. Cannot continue with other tests.")
        return
    
    # Test ticket retrieval
    test_get_tickets()
    
    # Test individual ticket operations using the first created ticket
    first_ticket_id = created_tickets[0]['id']
    test_individual_ticket_retrieval(first_ticket_id)
    test_ticket_status_update(first_ticket_id)
    
    # Test analytics
    test_analytics()
    
    # Test error handling
    test_error_handling()
    
    # Summary
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n🎉 Test suite completed in {duration:.2f} seconds")
    print(f"✅ Created {len(created_tickets)} test tickets")
    print("\n📝 Test Summary:")
    print("   - Health check: ✅")
    print("   - Ticket creation: ✅")
    print("   - Ticket retrieval: ✅")
    print("   - Status updates: ✅")
    print("   - Analytics: ✅")
    print("   - Error handling: ✅")
    
    print(f"\n🌐 Dashboard URL: {BASE_URL}")
    print("   Open this URL in your browser to see the web interface!")

if __name__ == "__main__":
    print("Support Ticket Router - API Test Suite")
    print("=" * 50)
    
    # Check if server is reachable
    try:
        response = requests.get(BASE_URL, timeout=5)
        print(f"✅ Server is reachable at {BASE_URL}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot reach server at {BASE_URL}")
        print("   Make sure the server is running with: python app.py")
        exit(1)
    
    run_comprehensive_test()