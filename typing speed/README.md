# 🏦 Banking Application with Behavioral Biometric Authentication

A comprehensive banking application that combines traditional banking features with advanced behavioral biometric authentication for enhanced security.

## 🚀 Features

### 🔐 Behavioral Authentication
- **Typing Pattern Analysis**: Analyzes keystroke dynamics and typing rhythm
- **Tap Pattern Recognition**: Captures touch patterns and tap timing
- **Navigation Behavior**: Tracks user navigation patterns and screen transitions
- **GPS Location Analysis**: Monitors location patterns for fraud detection
- **Swipe Pattern Analysis**: Analyzes touch gestures and swipe patterns

### 💳 Banking Features
- **Account Management**: Check balances, view transaction history
- **Money Transfer**: Bank transfers, UPI transfers, self transfers
- **Bill Payments**: Electricity, gas, water, internet, credit card bills
- **Mobile Recharge**: Prepaid and postpaid mobile recharges
- **DTH Recharge**: Cable and satellite TV recharges
- **Investment Services**: Fixed deposits, mutual funds, insurance
- **Loan Applications**: Personal, home, and vehicle loans
- **Cheque Services**: Request cheque books, stop cheques, check status
- **Statement Downloads**: Account and credit card statements

### 🛡️ Security Features
- **Multi-factor Behavioral Authentication**
- **Real-time Fraud Detection**
- **Session Management**
- **Transaction Verification**
- **Risk Assessment**

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Modern web browser

## 🛠️ Installation

1. **Clone or download the project files**

2. **Navigate to the project directory**
   ```bash
   cd "C:\Users\megha\OneDrive\Desktop\typing speed"
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python banking_auth_app.py
   ```

5. **Access the application**
   - Open your browser and go to: `http://localhost:8080`
   - The banking dashboard will be available at: `http://localhost:8080/banking`

## 🎯 Usage Guide

### Initial Setup
1. **Access the Application**: Navigate to `http://localhost:8080/banking`
2. **Behavioral Registration**: Complete the behavioral authentication setup
3. **Session Creation**: The system will create a secure session token

### Banking Operations

#### 🔄 Money Transfer
1. Click on "Bank Transfer" in Quick Actions
2. Select "Bank Transfer" from sub-menu
3. Fill in recipient details (Account number, IFSC, Amount)
4. Complete behavioral verification
5. Confirm transfer

#### 📱 UPI Transfer
1. Click on "UPI Transfer" in Quick Actions
2. Enter UPI ID or mobile number
3. Specify amount and message
4. Complete behavioral verification
5. Confirm payment

#### 📄 Bill Payments
1. Click on "Bills & Recharges" in Quick Actions
2. Select "Pay Bills" from sub-menu
3. Choose bill type (Electricity, Gas, etc.)
4. Enter biller ID and bill number
5. Complete behavioral verification
6. Confirm payment

#### 📱 Mobile Recharge
1. Click on "Bills & Recharges" in Quick Actions
2. Select "Mobile Recharge" from sub-menu
3. Enter mobile number and select operator
4. Specify recharge amount
5. Complete behavioral verification
6. Confirm recharge

#### 💰 Fixed Deposit
1. Click on "Investment" in Quick Actions
2. Select "Fixed Deposit" from sub-menu
3. Enter amount, tenure, and interest rate
4. Review maturity amount
5. Complete behavioral verification
6. Create FD

#### 🏦 Loan Applications
1. Click on "Loans" in Quick Actions
2. Select loan type (Personal, Home, Vehicle)
3. Fill in loan details and income information
4. Complete behavioral verification
5. Submit application

#### 📋 Cheque Services
1. Click on "Cheque Services" in Quick Actions
2. Select service (Request Cheque Book, Check Status, Stop Cheque)
3. Fill in required details
4. Complete behavioral verification
5. Submit request

#### 📊 Statements
1. Click on "Statements" in Quick Actions
2. Select statement type (Account, Credit Card)
3. Choose date range and format
4. Complete behavioral verification
5. Download statement

## 🔧 API Endpoints

### Authentication
- `POST /api/register` - User registration with behavioral data
- `POST /api/verify-session` - Session verification
- `POST /api/behavioral-auth` - Behavioral authentication

### Banking Operations
- `POST /api/banking/transfer` - Bank transfer
- `POST /api/banking/upi-transfer` - UPI transfer
- `POST /api/banking/bill-payment` - Bill payment
- `POST /api/banking/recharge` - Mobile recharge
- `GET /api/banking/account-balance` - Get account balance
- `GET /api/banking/transaction-history` - Get transaction history
- `POST /api/banking/cheque-request` - Request cheque book
- `POST /api/banking/fd-creation` - Create fixed deposit
- `POST /api/banking/loan-application` - Apply for loan

### Behavioral Analysis
- `POST /api/analyze-typing` - Analyze typing patterns
- `POST /api/analyze-tap` - Analyze tap patterns
- `POST /api/analyze-navigation` - Analyze navigation patterns
- `POST /api/analyze-gps` - Analyze GPS patterns
- `POST /api/analyze-swipe` - Analyze swipe patterns

## 🧪 Testing

### Manual Testing
1. **UI Testing**: Test all banking features through the web interface
2. **Behavioral Testing**: Test behavioral authentication with different patterns
3. **Security Testing**: Test session management and authentication
4. **API Testing**: Test all API endpoints using tools like Postman

### Automated Testing
```bash
# Run tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test file
pytest test_banking_auth.py
```

## 🔒 Security Considerations

### Behavioral Biometrics
- **Data Privacy**: All behavioral data is encrypted and stored securely
- **Pattern Analysis**: Uses machine learning for pattern recognition
- **Risk Assessment**: Real-time risk scoring for transactions
- **Fraud Detection**: Advanced algorithms for fraud detection

### Session Management
- **Secure Tokens**: JWT-based session tokens
- **Token Expiration**: Automatic token expiration
- **Session Validation**: Continuous session validation
- **Logout**: Secure session termination

### Transaction Security
- **Multi-factor Verification**: Behavioral + traditional authentication
- **Transaction Limits**: Configurable transaction limits
- **Audit Trail**: Complete transaction audit trail
- **Real-time Monitoring**: Continuous transaction monitoring

## 🚀 Deployment

### Development
```bash
python banking_auth_app.py
```

### Production
```bash
# Using Gunicorn
gunicorn banking_auth_app:app -w 4 -k uvicorn.workers.UvicornWorker

# Using Uvicorn
uvicorn banking_auth_app:app --host 0.0.0.0 --port 8080
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8080

CMD ["uvicorn", "banking_auth_app:app", "--host", "0.0.0.0", "--port", "8080"]
```

## 📊 Monitoring and Logging

### Application Monitoring
- **Performance Metrics**: Response times, throughput
- **Error Tracking**: Error rates and types
- **User Analytics**: User behavior and patterns
- **Security Events**: Authentication and fraud events

### Logging
- **Structured Logging**: JSON-formatted logs
- **Log Levels**: DEBUG, INFO, WARNING, ERROR
- **Log Rotation**: Automatic log rotation
- **Centralized Logging**: Centralized log management

## 🔧 Configuration

### Environment Variables
```bash
# Server Configuration
PORT=8080
HOST=0.0.0.0

# Security Configuration
SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256
JWT_EXPIRATION=3600

# Database Configuration (if using database)
DATABASE_URL=sqlite:///banking.db

# Behavioral Analysis Configuration
BEHAVIORAL_THRESHOLD=0.8
PATTERN_MEMORY_SIZE=100
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new features
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Contact the development team
- Check the documentation

## 🔄 Updates and Maintenance

### Regular Updates
- **Security Patches**: Regular security updates
- **Feature Updates**: New banking features
- **Performance Improvements**: Optimization updates
- **Bug Fixes**: Bug fixes and improvements

### Maintenance Schedule
- **Daily**: Log analysis and monitoring
- **Weekly**: Performance review and optimization
- **Monthly**: Security audit and updates
- **Quarterly**: Feature updates and improvements

---

**⚠️ Important Notes:**
- This is a demonstration application for educational purposes
- In production, implement proper security measures
- Use HTTPS in production environments
- Implement proper database security
- Follow banking regulations and compliance requirements
- Regular security audits are recommended 