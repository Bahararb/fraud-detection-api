# Fraud Detection ML API

## Overview
This project is a machine learning-based fraud detection system deployed on AWS Elastic Beanstalk using Docker and FastAPI. The system predicts whether a transaction is fraudulent based on input features.

---

## Project Features
- Machine Learning model trained on credit card fraud dataset
- REST API built using FastAPI
- Containerized using Docker
- Deployed on AWS Elastic Beanstalk
- Real-time prediction endpoint

---

## Run Locally (Docker)

### 1. Build Docker image
```bash
docker build -t fraud-api .