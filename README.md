# Candle-Light

AI-Powered Candlestick Pattern Analysis Platform

---

## 1. The Problem
In the fast-paced world of financial trading, objective analysis is often clouded by human emotion and bias. Traders, especially beginners, struggle to consistently identify complex candlestick patterns across varying timeframes. Manual analysis is time-consuming, prone to error, and often leads to missed opportunities or poor decision-making based on subjective interpretation rather than data-driven evidence.

## 2. The Solution
Candle-Light is an automated, AI-powered platform designed to democratize technical analysis. By leveraging advanced Vision Language Models (VLMs), the application instantly analyzes chart images to identify candlestick patterns, detect market bias, and provide confidence scores. This allows traders to validate their own analysis, reduce emotional bias, and make informed decisions based on objective, algorithmic insights.

## 3. Current Status
We have successfully developed a robust, full-stack web application that serves as the foundation for this vision.
- **Backend**: A high-performance FastAPI server is capable of handling image uploads, managing user sessions via secure JWT authentication, and storing analysis history in a relational database.
- **Frontend**: A responsive, modern React interface allowing users to easily upload charts, view detailed analysis results, and track their history.
- **AI Integration**: The system boasts a modular AI service layer ready to connect with industry-leading models like OpenAI's GPT-4o and Google's Gemini for deep visual reasoning.

## 4. System Design & Architecture
Our architecture follows a Microservices-inspired Monolithic approach, balancing simplicity with scalability.

### Frontend
- **Framework**: React 18 with Vite for lightning-fast performance.
- **Language**: TypeScript for type safety and code reliability.
- **Styling**: Tailwind CSS and shadcn/ui for a premium, accessible design system.
- **State Management**: React Query for efficient server state synchronization.

### Backend
- **Framework**: FastAPI (Python) for asynchronous, high-concurrency request handling.
- **Database**: PostgreSQL (Production) / SQLite (Development) using SQLAlchemy ORM for robust data management.
- **Authentication**: OAuth2 and JWT (JSON Web Tokens) for secure, stateless user authentication.
- **AI Layer**: Abstracted service interfaces to support multiple AI providers (OpenAI, Gemini) interchangeably.

## 5. Development Roadmap
We are currently in the **Integration Phase**, moving towards a public beta release.

- **Phase 1 (Completed)**: Core backend structure, database design, and frontend UI implementation.
- **Phase 2 (Completed)**: Frontend-Backend integration, Authentication flow, and Analysis API endpoints.
- **Phase 3 (Current)**: Refinement of AI prompt engineering and confidence scoring algorithms.

## 6. Future Scope
Our vision extends beyond simple pattern recognition. We aim to build a comprehensive trading companion.
- **Real-Time Streaming**: Integration with WebSocket APIs to analyze live market data feeds.
- **Multi-Frame Analysis**: Capability to correlate patterns across different timeframes (e.g., 1H, 4H, Daily) for stronger signals.
- **Social Trading**: Community features enabling users to share analysis, vote on pattern validity, and learn from others.
- **Mobile Application**: Native mobile experience for on-the-go analysis.
- **Backtesting Engine**: Ability to simulate identified patterns against historical data to verify profitability.

---

## Installation & Running

### Backend
```bash
cd backend
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```
