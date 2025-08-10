# Module 6: Practical RAG Implementation Workshop

**Date:** August 21, 2025  
**Duration:** Full Day (09:30 - 16:00)  
**Instructor:** Mr. Amornpan Phornchaichareon (National Telecom Public Company Limited)

## 📖 Workshop Overview

This intensive full-day workshop is the practical culmination of the course, where participants build complete, production-ready RAG systems from scratch. Through hands-on implementation, real-world testing, and collaborative projects, participants will develop comprehensive RAG solutions specifically tailored for telecommunications applications.

## 🎯 Workshop Objectives

By the end of this workshop, participants will be able to:
- **Build** complete RAG systems from the ground up using pre-configured environments
- **Implement** sophisticated customer service chatbots with telecommunications-specific features
- **Test and validate** LLM applications against real telecommunication scenarios
- **Customize** RAG systems for specific organizational use cases
- **Present** technical solutions effectively to stakeholders and peers
- **Collaborate** on complex technical projects with distributed teams

---

## 🌅 Morning Session (09:30 - 12:00): Foundation Building

**Duration:** 2.5 hours  
**Focus:** Infrastructure setup and core implementation

### 1. Pre-configured Google Colab Notebooks Setup (30 minutes)

#### 1.1 Environment Initialization and Validation

##### **Workshop Environment Setup**
```python
# Workshop initialization script - Cell 1
print("🚀 APT LLMs for Telecom - Day 3 Workshop")
print("=" * 50)

# Verify Google Colab environment
import sys
print(f"Python version: {sys.version}")
print(f"Running on: {'Google Colab' if 'google.colab' in sys.modules else 'Local environment'}")

# Check GPU availability
import torch
if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("CPU-only environment")

# Set random seeds for reproducibility
import random
import numpy as np
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
print("✅ Environment initialized successfully!")
```

##### **Required Package Installation with Version Control**
```python
# Cell 2: Install and verify all required packages
packages_to_install = [
    "sentence-transformers==2.2.2",
    "chromadb==0.4.15",
    "langchain==0.0.350",
    "langchain-community==0.0.38",
    "openai==1.3.0",
    "gradio==4.8.0",
    "streamlit==1.28.0",
    "pandas==2.0.3",
    "numpy==1.24.3",
    "matplotlib==3.7.2",
    "seaborn==0.12.2",
    "scikit-learn==1.3.0",
    "tqdm==4.66.1",
    "requests==2.31.0",
    "beautifulsoup4==4.12.2"
]

import subprocess
import sys

def install_package(package):
    \"\"\"Install package with error handling\"\"\"
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
        print(f"✅ {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

print("Installing required packages...")
success_count = 0
for package in packages_to_install:
    if install_package(package):
        success_count += 1

print(f"\\n📦 Installation complete: {success_count}/{len(packages_to_install)} packages installed")

# Verify critical imports
try:
    import sentence_transformers
    import chromadb
    import langchain
    import gradio
    import pandas as pd
    import numpy as np
    print("✅ All critical packages imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
```

##### **Telecommunications Workshop Dataset Preparation**
```python
# Cell 3: Load comprehensive telecommunications dataset
telecommunications_knowledge_base = {
    "customer_service": [
        {
            "id": "cs_001",
            "title": "Mobile Plan Comparison and Selection",
            "content": \"\"\"Our mobile plans are designed to meet diverse customer needs:

BASIC PLAN ($25/month):
- 2GB high-speed data
- Unlimited domestic calls and texts
- Voicemail and caller ID included
- No contract required
- International calling: $0.05/minute

STANDARD PLAN ($45/month):
- 10GB high-speed data
- Unlimited domestic calls and texts
- 5GB mobile hotspot
- HD video streaming included
- International calling: $0.03/minute
- Free roaming in Canada and Mexico

PREMIUM PLAN ($65/month):
- Unlimited high-speed data
- 5G network access where available
- 15GB mobile hotspot
- 4K video streaming included
- International calling: $0.02/minute
- Free roaming in 50+ countries
- Priority network access during busy periods

FAMILY PLANS:
- 2 lines: Save $10/month per line
- 3+ lines: Save $15/month per line
- Shared data pools available
- Individual usage monitoring

STUDENT DISCOUNT:
- 20% off any plan with valid student ID
- Available for high school and college students
- Verification required annually\"\"\",
            "category": "billing",
            "subcategory": "plans",
            "tags": ["mobile", "pricing", "features", "family", "student"],
            "last_updated": "2024-01-15",
            "confidence": "high"
        },
        {
            "id": "cs_002", 
            "title": "Internet Service Troubleshooting Guide",
            "content": \"\"\"Step-by-step internet connectivity troubleshooting:

SLOW INTERNET SPEEDS:
1. Run speed test at speedtest.net
2. Expected speeds: Basic (25 Mbps), Standard (100 Mbps), Premium (500 Mbps)
3. Check for bandwidth-heavy applications (streaming, downloads)
4. Restart modem and router (power cycle for 30 seconds)
5. Check for interference from other devices
6. Update router firmware if needed

NO INTERNET CONNECTION:
1. Check all cable connections (power, ethernet, coaxial)
2. Look for service outages at status.telecom.com
3. Restart modem: unplug for 30 seconds, reconnect
4. Restart router: unplug for 30 seconds, reconnect
5. Check device WiFi settings and password
6. Try ethernet connection directly to modem

INTERMITTENT CONNECTION:
1. Monitor connection stability over 24 hours
2. Check for overheating (ensure proper ventilation)
3. Inspect cables for damage or loose connections
4. Update network drivers on connected devices
5. Consider router replacement if over 5 years old

WIFI SPECIFIC ISSUES:
1. Check WiFi password and network name (SSID)
2. Move closer to router (optimal range: 30 feet)
3. Reduce interference: avoid microwaves, baby monitors
4. Switch WiFi channels in router settings
5. Consider WiFi extender for large homes

If problems persist after these steps, contact technical support at 1-800-TELECOM or schedule a technician visit online.\"\"\",
            "category": "technical_support",
            "subcategory": "internet",
            "tags": ["troubleshooting", "wifi", "connection", "speed"],
            "last_updated": "2024-01-10",
            "confidence": "high"
        },
        {
            "id": "cs_003",
            "title": "Bill Understanding and Payment Options",
            "content": \"\"\"Complete guide to understanding your telecommunications bill:

BILL SECTIONS EXPLAINED:
1. Account Summary: Total amount due, payment due date, previous balance
2. Current Charges: Monthly service fees, equipment rental, taxes
3. Usage Charges: Overage fees, international calls, premium services
4. Credits and Adjustments: Discounts, refunds, promotional credits
5. Regulatory Fees: Government-mandated charges and taxes

PAYMENT METHODS:
- Online: Account portal at www.telecom.com/pay (free)
- Phone: Automated system 1-800-PAY-BILL (free)
- Mobile App: iOS and Android apps available (free)
- Auto-Pay: Automatic monthly deduction ($5 discount)
- Mail: Check or money order to PO Box 12345 (allow 5-7 days)
- Retail Locations: 500+ authorized payment centers

PAYMENT SCHEDULES:
- Due date: 30 days from bill date
- Grace period: 10 days past due date
- Late fee: $25 after grace period
- Service suspension: 45 days past due
- Reconnection fee: $50 for suspended service

BILLING CYCLES:
- Bills generated on same date each month
- Service period: Previous month's usage
- Paper bills: $3 monthly fee (waived with auto-pay)
- Electronic bills: Free via email or app

DISPUTE PROCESS:
1. Review bill carefully within 30 days
2. Contact customer service: 1-800-CUSTOMER
3. Formal dispute: Submit within 60 days
4. Investigation period: 30 days maximum
5. Resolution notification via mail or email\"\"\",
            "category": "billing",
            "subcategory": "payment",
            "tags": ["bill", "payment", "due_date", "fees", "dispute"],
            "last_updated": "2024-01-20",
            "confidence": "high"
        },
        {
            "id": "cs_004",
            "title": "5G Network Coverage and Device Compatibility", 
            "content": \"\"\"Comprehensive 5G network information:

5G COVERAGE AREAS:
- Ultra Wideband 5G: 200+ cities and growing
- Extended Range 5G: 2,500+ cities and towns
- Coverage map: Available at www.telecom.com/coverage
- Real-time network status: Network monitoring app

5G SPEED EXPECTATIONS:
- Ultra Wideband: 100-1000 Mbps typical
- Extended Range: 50-200 Mbps typical
- Factors affecting speed: Device, location, network traffic
- Speed tests: Use official Telecom Speed Test app

COMPATIBLE DEVICES:
iPhone Models:
- iPhone 15 series (all models)
- iPhone 14 series (all models) 
- iPhone 13 series (all models)
- iPhone 12 series (all models)
- iPhone SE 3rd generation (2022)

Android Devices:
- Samsung Galaxy S24, S23, S22 series
- Google Pixel 8, 7, 6 series
- OnePlus 12, 11, 10 series
- Motorola Edge series
- Complete list: www.telecom.com/5g-devices

5G PLAN REQUIREMENTS:
- Premium plan required for Ultra Wideband access
- Standard plan includes Extended Range 5G
- No additional fees for 5G access
- International 5G roaming in 25+ countries

TROUBLESHOOTING 5G:
1. Verify device compatibility
2. Ensure 5G is enabled in device settings
3. Check coverage in your area
4. Restart device if experiencing issues
5. Update device software regularly
6. Contact support for persistent issues\"\"\",
            "category": "network_services",
            "subcategory": "5g",
            "tags": ["5g", "coverage", "devices", "speed", "compatibility"],
            "last_updated": "2024-01-25",
            "confidence": "high"
        },
        {
            "id": "cs_005",
            "title": "Business Services and Enterprise Solutions",
            "content": \"\"\"Complete business telecommunications solutions:

SMALL BUSINESS PACKAGES:
Starter Business ($75/month):
- 5 lines with unlimited talk and text
- 10GB shared data pool
- Basic business tools and apps
- Email support and online portal
- Standard installation included

Professional Business ($150/month):
- 10 lines with unlimited talk and text
- 50GB shared data pool
- Advanced business applications
- Priority customer support
- Mobile device management
- International calling included

Enterprise Business (Custom pricing):
- Unlimited lines and data
- Dedicated account management
- 24/7 technical support
- Custom network solutions
- Advanced security features
- Service level agreements (SLAs)

ADDITIONAL BUSINESS SERVICES:
- Dedicated Internet Access (DIA)
- MPLS networking solutions
- Cloud PBX phone systems
- Video conferencing solutions
- Cybersecurity services
- IoT connectivity solutions

FIBER INTERNET FOR BUSINESS:
- Speeds up to 10 Gbps
- 99.9% uptime guarantee
- Symmetric upload/download speeds
- Dedicated customer support
- Professional installation
- Scalable bandwidth options

SUPPORT SERVICES:
- Business account portal
- Usage analytics and reporting
- Employee device management
- Technical support: 1-800-BIZ-HELP
- On-site support available
- Training and consultation services\"\"\",
            "category": "business_services",
            "subcategory": "packages",
            "tags": ["business", "enterprise", "fiber", "support", "pricing"],
            "last_updated": "2024-01-18",
            "confidence": "high"
        }
    ],
    "technical_documentation": [
        {
            "id": "tech_001",
            "title": "Router Configuration and WiFi Optimization",
            "content": \"\"\"Advanced router setup and optimization guide:

INITIAL ROUTER SETUP:
1. Connect router to modem via Ethernet cable
2. Power on router and wait for startup (2-3 minutes)
3. Connect device to router's default WiFi network
4. Open web browser and navigate to 192.168.1.1
5. Login with default credentials (usually admin/admin)
6. Follow setup wizard for internet configuration

WIFI OPTIMIZATION SETTINGS:
Network Name (SSID):
- Choose unique, identifiable name
- Avoid personal information
- Example: "YourName_Home_5G" and "YourName_Home_2.4G"

Password Security:
- WPA3 encryption (preferred) or WPA2
- Minimum 12 characters
- Mix of letters, numbers, symbols
- Avoid dictionary words

Channel Selection:
2.4GHz Band:
- Channels 1, 6, 11 (non-overlapping)
- Use WiFi analyzer to check interference
- Auto-channel may not always be optimal

5GHz Band:
- More channels available (36, 40, 44, 48, 149, 153, 157, 161)
- Less congested than 2.4GHz
- Shorter range but higher speeds

ADVANCED SETTINGS:
Quality of Service (QoS):
- Prioritize gaming and video streaming
- Limit bandwidth for specific devices
- Set upload/download priorities

Guest Network:
- Separate network for visitors
- Bandwidth limitations available
- Automatic disconnect options

Firmware Updates:
- Check monthly for updates
- Enable automatic updates if available
- Backup settings before updating

TROUBLESHOOTING TIPS:
- Restart router monthly
- Check for overheating
- Update device drivers
- Position router centrally and elevated
- Avoid interference from appliances\"\"\",
            "category": "technical_support", 
            "subcategory": "equipment",
            "tags": ["router", "wifi", "setup", "optimization", "security"],
            "last_updated": "2024-01-12",
            "confidence": "high"
        },
        {
            "id": "tech_002",
            "title": "Mobile Device APN Settings and Data Configuration",
            "content": \"\"\"Complete APN configuration guide for all devices:

AUTOMATIC APN CONFIGURATION:
1. Insert SIM card into device
2. Power on device and wait for network detection
3. Settings should configure automatically
4. If not automatic, use manual configuration below

MANUAL APN SETTINGS:
For Android Devices:
- Settings → Network & Internet → Mobile Network
- Access Point Names → Add new APN
- Name: Telecom Internet
- APN: internet.telecom.com
- Username: (leave blank)
- Password: (leave blank)
- Authentication Type: None
- APN Type: default,supl,mms

For iPhone:
- Settings → Cellular → Cellular Data Options
- Cellular Network → Automatic (toggle off if needed)
- APN: internet.telecom.com
- Username: (leave blank)
- Password: (leave blank)

MMS SETTINGS:
- MMS APN: mms.telecom.com
- MMS Proxy: 10.10.10.10
- MMS Port: 80
- MMS Max Message Size: 1048576

DATA ROAMING SETTINGS:
Domestic Roaming:
- Usually automatic
- No additional charges
- Contact support if issues persist

International Roaming:
- Enable in device settings
- Check rates: www.telecom.com/international
- Consider international plans before travel
- Monitor usage to avoid overage charges

TROUBLESHOOTING DATA ISSUES:
1. Verify APN settings match exactly
2. Restart device after configuration
3. Check for carrier settings updates
4. Ensure data plan is active
5. Contact customer service for activation issues

NETWORK MODE SELECTION:
- Automatic (recommended)
- LTE/4G preferred
- 3G only (for older devices)
- 2G only (emergency use)\"\"\",
            "category": "technical_support",
            "subcategory": "mobile",
            "tags": ["apn", "mobile", "data", "configuration", "mms"],
            "last_updated": "2024-01-14",
            "confidence": "high"
        }
    ],
    "policies_procedures": [
        {
            "id": "policy_001",
            "title": "Privacy Policy and Data Protection",
            "content": \"\"\"Comprehensive privacy and data protection policy:

DATA COLLECTION PRACTICES:
Information We Collect:
- Account information: Name, address, phone number, email
- Service usage: Call logs, data usage, messaging records
- Device information: Model, operating system, unique identifiers
- Location data: General location for service delivery
- Payment information: Billing address, payment methods

How We Use Your Information:
- Service provision and billing
- Customer support and troubleshooting
- Network optimization and maintenance
- Fraud prevention and security
- Marketing communications (with consent)
- Legal compliance and regulatory requirements

CUSTOMER RIGHTS:
Access and Control:
- View your personal data via account portal
- Download your data in portable format
- Correct inaccurate information
- Delete account and associated data
- Opt-out of marketing communications

Data Retention:
- Active account data: Retained while account is active
- Billing records: 7 years for tax purposes
- Call detail records: 18 months maximum
- Marketing preferences: Until you opt-out
- Legal hold data: As required by law

THIRD-PARTY SHARING:
We Share Information With:
- Service providers for technical operations
- Payment processors for billing
- Credit agencies for account approval
- Law enforcement when legally required
- Emergency services for 911 calls

We Do NOT Share:
- Personal data for marketing by others
- Call content or message content
- Browsing history or app usage
- Location data beyond service requirements

SECURITY MEASURES:
Technical Safeguards:
- Encryption of data in transit and at rest
- Multi-factor authentication for accounts
- Regular security audits and assessments
- Employee training on data protection
- Incident response procedures

GDPR AND CCPA COMPLIANCE:
- Right to know what data we collect
- Right to delete personal information
- Right to opt-out of data sales (we don't sell data)
- Right to non-discrimination
- Designated privacy officer contact

CONTACT FOR PRIVACY CONCERNS:
- Email: privacy@telecom.com
- Phone: 1-800-PRIVACY
- Mail: Privacy Officer, PO Box 98765
- Response time: 30 days maximum\"\"\",
            "category": "policy",
            "subcategory": "privacy",
            "tags": ["privacy", "gdpr", "ccpa", "data_protection", "rights"],
            "last_updated": "2024-01-30",
            "confidence": "high"
        }
    ]
}

# Convert to DataFrame for processing
all_documents = []
for category, docs in telecommunications_knowledge_base.items():
    all_documents.extend(docs)

documents_df = pd.DataFrame(all_documents)
print(f"📊 Loaded {len(documents_df)} telecommunications documents")
print(f"Categories: {documents_df['category'].unique()}")
print(f"Subcategories: {documents_df['subcategory'].unique()}")
documents_df.head()
```

#### 1.2 Workshop-Specific Configuration and Tools

##### **Advanced RAG System Class for Workshop**
```python
# Cell 4: Workshop RAG system implementation
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import re

class TelecomWorkshopRAG:
    \"\"\"
    Advanced RAG system specifically designed for telecommunications workshop
    \"\"\"
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 600,
                 chunk_overlap: int = 100):
        
        # Initialize components
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Load models and tools
        self._initialize_components()
        
        # Conversation tracking
        self.conversation_history = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Performance metrics
        self.query_count = 0
        self.response_times = []
        
    def _initialize_components(self):
        \"\"\"Initialize all RAG components\"\"\"
        print("🔧 Initializing RAG components...")
        
        # Embedding model
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer(self.embedding_model_name)
        print(f"✅ Loaded embedding model: {self.embedding_model_name}")
        
        # Text splitter
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\\n\\n", "\\n", ". ", " ", ""]
        )
        print("✅ Initialized text splitter")
        
        # Vector database
        import chromadb
        self.chroma_client = chromadb.Client()
        self.collection = None
        print("✅ Initialized Chroma client")
        
        # Query preprocessing tools
        self.telecom_synonyms = {
            "internet": ["wifi", "broadband", "connection", "online"],
            "mobile": ["cell phone", "cellular", "phone", "device"],
            "bill": ["invoice", "payment", "charge", "fee"],
            "plan": ["package", "service", "subscription"],
            "5g": ["fifth generation", "5g network", "high speed"],
            "support": ["help", "assistance", "customer service"]
        }
        print("✅ Loaded telecommunications vocabulary")
        
    def preprocess_documents(self, documents_df: pd.DataFrame) -> List[Dict]:
        \"\"\"Advanced document preprocessing with telecommunications optimization\"\"\"
        print("📄 Processing telecommunications documents...")
        
        processed_chunks = []
        
        for idx, row in documents_df.iterrows():
            # Enhanced document formatting
            document_text = f\"\"\"Title: {row['title']}
Category: {row['category']} | Subcategory: {row['subcategory']}
Last Updated: {row['last_updated']}
Tags: {', '.join(row['tags']) if isinstance(row['tags'], list) else row['tags']}

Content:
{row['content']}\"\"\"
            
            # Split into semantic chunks
            chunks = self.text_splitter.split_text(document_text)
            
            for chunk_idx, chunk in enumerate(chunks):
                # Clean and enhance chunk
                cleaned_chunk = self._clean_text(chunk)
                
                chunk_metadata = {
                    'document_id': row['id'],
                    'chunk_id': f"{row['id']}_chunk_{chunk_idx}",
                    'title': row['title'],
                    'category': row['category'],
                    'subcategory': row['subcategory'],
                    'tags': row['tags'] if isinstance(row['tags'], list) else [row['tags']],
                    'confidence': row['confidence'],
                    'last_updated': row['last_updated'],
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks)
                }
                
                processed_chunks.append({
                    'text': cleaned_chunk,
                    'metadata': chunk_metadata
                })
        
        print(f"✅ Processed {len(processed_chunks)} document chunks")
        return processed_chunks
    
    def _clean_text(self, text: str) -> str:
        \"\"\"Clean and normalize text for better processing\"\"\"
        # Remove extra whitespace
        text = re.sub(r'\\s+', ' ', text)
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\\w\\s.,;:!?\\-$()%/]', ' ', text)
        # Normalize currency and numbers
        text = re.sub(r'\\$([0-9,]+)', r'\\1 dollars', text)
        return text.strip()
    
    def create_enhanced_vector_database(self, processed_chunks: List[Dict]):
        \"\"\"Create optimized vector database for telecommunications\"\"\"
        print("🗄️ Creating enhanced vector database...")
        
        # Create collection with optimized settings
        collection_name = f"telecom_workshop_{self.session_id}"
        
        self.collection = self.chroma_client.create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:M": 16,
                "hnsw:ef_construction": 200,
                "description": "Telecommunications knowledge base for workshop"
            }
        )
        
        # Prepare data for insertion
        texts = [chunk['text'] for chunk in processed_chunks]
        metadatas = [chunk['metadata'] for chunk in processed_chunks]
        ids = [chunk['metadata']['chunk_id'] for chunk in processed_chunks]
        
        # Generate embeddings in batches
        batch_size = 32
        all_embeddings = []
        
        print("🔄 Generating embeddings...")
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embedder.encode(batch_texts)
            all_embeddings.extend(batch_embeddings.tolist())
        
        # Insert into vector database
        self.collection.add(
            documents=texts,
            embeddings=all_embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Created vector database with {len(texts)} chunks")
        
    def enhance_query(self, query: str) -> str:
        \"\"\"Enhance query with telecommunications synonyms and context\"\"\"
        enhanced_query = query.lower()
        
        # Expand with synonyms
        for term, synonyms in self.telecom_synonyms.items():
            if term in enhanced_query:
                enhanced_query += f" {' '.join(synonyms)}"
        
        # Add context from conversation history
        if self.conversation_history:
            recent_context = " ".join([
                entry['query'] for entry in self.conversation_history[-2:]
            ])
            enhanced_query = f"Context: {recent_context} Current query: {enhanced_query}"
        
        return enhanced_query
    
    def intelligent_retrieval(self, 
                            query: str, 
                            top_k: int = 5,
                            similarity_threshold: float = 0.7) -> Dict:
        \"\"\"Advanced retrieval with filtering and ranking\"\"\"
        
        start_time = time.time()
        
        # Enhance query
        enhanced_query = self.enhance_query(query)
        
        # Generate query embedding
        query_embedding = self.embedder.encode([enhanced_query])
        
        # Retrieve with higher initial count for filtering
        initial_results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=min(top_k * 3, 15),  # Get more for filtering
            include=['documents', 'metadatas', 'distances']
        )
        
        # Filter by similarity threshold and relevance
        filtered_results = {
            'documents': [],
            'metadatas': [],
            'distances': [],
            'relevance_scores': []
        }
        
        for doc, meta, distance in zip(
            initial_results['documents'][0],
            initial_results['metadatas'][0], 
            initial_results['distances'][0]
        ):
            similarity = 1 - distance  # Convert distance to similarity
            
            if similarity >= similarity_threshold:
                # Calculate relevance score
                relevance_score = self._calculate_relevance(query, doc, meta)
                
                filtered_results['documents'].append(doc)
                filtered_results['metadatas'].append(meta)
                filtered_results['distances'].append(distance)
                filtered_results['relevance_scores'].append(relevance_score)
        
        # Sort by combined similarity and relevance
        if filtered_results['documents']:
            combined_scores = [
                0.7 * (1 - dist) + 0.3 * rel_score
                for dist, rel_score in zip(
                    filtered_results['distances'],
                    filtered_results['relevance_scores']
                )
            ]
            
            # Sort by combined score (descending)
            sorted_indices = sorted(
                range(len(combined_scores)), 
                key=lambda i: combined_scores[i], 
                reverse=True
            )
            
            # Reorder results
            for key in filtered_results:
                filtered_results[key] = [
                    filtered_results[key][i] for i in sorted_indices[:top_k]
                ]
        
        # Record performance
        retrieval_time = time.time() - start_time
        self.response_times.append(retrieval_time)
        
        return {
            'results': filtered_results,
            'retrieval_time': retrieval_time,
            'total_candidates': len(initial_results['documents'][0]),
            'filtered_count': len(filtered_results['documents'])
        }
    
    def _calculate_relevance(self, query: str, document: str, metadata: Dict) -> float:
        \"\"\"Calculate document relevance score based on multiple factors\"\"\"
        relevance_score = 0.0
        
        # Category relevance
        query_lower = query.lower()
        category_keywords = {
            'billing': ['bill', 'payment', 'charge', 'fee', 'plan', 'cost'],
            'technical_support': ['problem', 'issue', 'troubleshoot', 'fix', 'help'],
            'network_services': ['5g', '4g', 'network', 'coverage', 'speed'],
            'business_services': ['business', 'enterprise', 'commercial']
        }
        
        category = metadata.get('category', '')
        if category in category_keywords:
            keyword_matches = sum(1 for keyword in category_keywords[category] 
                                if keyword in query_lower)
            relevance_score += keyword_matches * 0.1
        
        # Confidence score from metadata
        confidence = metadata.get('confidence', 'medium')
        confidence_scores = {'high': 0.3, 'medium': 0.2, 'low': 0.1}
        relevance_score += confidence_scores.get(confidence, 0.1)
        
        # Recency (newer documents get slight boost)
        last_updated = metadata.get('last_updated', '2020-01-01')
        try:
            update_date = datetime.strptime(last_updated, '%Y-%m-%d')
            days_old = (datetime.now() - update_date).days
            recency_score = max(0, 0.2 - (days_old / 365) * 0.1)  # Decay over year
            relevance_score += recency_score
        except:
            pass
        
        return min(relevance_score, 1.0)  # Cap at 1.0

# Initialize the workshop RAG system
print("🚀 Initializing Telecommunications Workshop RAG System...")
workshop_rag = TelecomWorkshopRAG()
```

### 2. Building Knowledge Base with Telecommunications Documents (45 minutes)

#### 2.1 Document Processing and Optimization

##### **Comprehensive Document Processing Pipeline**
```python
# Cell 5: Advanced document processing
def process_telecommunications_knowledge_base():
    \"\"\"Process the complete telecommunications knowledge base\"\"\"
    
    print("📚 Processing Telecommunications Knowledge Base")
    print("=" * 60)
    
    # Process documents with advanced chunking
    processed_chunks = workshop_rag.preprocess_documents(documents_df)
    
    # Analyze processing results
    category_distribution = {}
    for chunk in processed_chunks:
        category = chunk['metadata']['category']
        category_distribution[category] = category_distribution.get(category, 0) + 1
    
    print("\\n📊 Document Processing Statistics:")
    print(f"Total documents: {len(documents_df)}")
    print(f"Total chunks: {len(processed_chunks)}")
    print(f"Average chunks per document: {len(processed_chunks) / len(documents_df):.1f}")
    
    print("\\n📂 Category Distribution:")
    for category, count in category_distribution.items():
        print(f"  {category}: {count} chunks")
    
    # Create enhanced vector database
    workshop_rag.create_enhanced_vector_database(processed_chunks)
    
    return processed_chunks

# Process the knowledge base
processed_chunks = process_telecommunications_knowledge_base()
```

##### **Knowledge Base Quality Assessment**
```python
# Cell 6: Quality assessment and validation
def assess_knowledge_base_quality():
    \"\"\"Assess the quality and coverage of the knowledge base\"\"\"
    
    print("🔍 Knowledge Base Quality Assessment")
    print("=" * 50)
    
    # Test queries for coverage assessment
    test_queries = [
        "What mobile plans do you offer?",
        "How do I troubleshoot my internet connection?",
        "How can I pay my bill?",
        "What is 5G and where is it available?",
        "Do you offer business services?",
        "How do I set up my router?",
        "What are your privacy policies?",
        "How do I configure APN settings?"
    ]
    
    coverage_results = []
    
    for query in test_queries:
        print(f"\\n🔍 Testing: '{query}'")
        
        retrieval_result = workshop_rag.intelligent_retrieval(
            query, 
            top_k=3, 
            similarity_threshold=0.5
        )
        
        results = retrieval_result['results']
        
        if results['documents']:
            top_similarity = 1 - results['distances'][0]
            top_category = results['metadatas'][0]['category']
            
            print(f"  ✅ Found {len(results['documents'])} relevant chunks")
            print(f"  📊 Top similarity: {top_similarity:.3f}")
            print(f"  📂 Category: {top_category}")
            
            coverage_results.append({
                'query': query,
                'found': True,
                'similarity': top_similarity,
                'category': top_category,
                'chunk_count': len(results['documents'])
            })
        else:
            print(f"  ❌ No relevant content found")
            coverage_results.append({
                'query': query,
                'found': False,
                'similarity': 0.0,
                'category': None,
                'chunk_count': 0
            })
    
    # Calculate coverage statistics
    coverage_rate = sum(1 for r in coverage_results if r['found']) / len(coverage_results)
    avg_similarity = sum(r['similarity'] for r in coverage_results if r['found']) / max(1, sum(1 for r in coverage_results if r['found']))
    
    print(f"\\n📈 Knowledge Base Coverage Statistics:")
    print(f"Coverage rate: {coverage_rate:.1%}")
    print(f"Average similarity: {avg_similarity:.3f}")
    print(f"Total test queries: {len(test_queries)}")
    
    return coverage_results

# Assess knowledge base quality
coverage_assessment = assess_knowledge_base_quality()
```

### 3. Implementing Vector Embeddings using Free-tier Services (45 minutes)

#### 3.1 Advanced Embedding Strategies and Optimization

##### **Multi-Model Embedding Comparison**
```python
# Cell 7: Compare different embedding models
def compare_embedding_models():
    \"\"\"Compare different embedding models for telecommunications content\"\"\"
    
    print("🔬 Embedding Model Comparison for Telecommunications")
    print("=" * 60)
    
    # Test models (using free/open-source options)
    models_to_test = [
        "all-MiniLM-L6-v2",           # Fast, good general performance
        "all-mpnet-base-v2",          # Higher quality, slower
        "paraphrase-multilingual-MiniLM-L12-v2"  # Multilingual support
    ]
    
    # Test query for comparison
    test_query = "How much do mobile data plans cost?"
    test_document = documents_df.iloc[0]['content'][:500]  # First 500 chars
    
    embedding_results = {}
    
    for model_name in models_to_test:
        print(f"\\n🔄 Testing {model_name}...")
        
        try:
            # Load model
            model = SentenceTransformer(model_name)
            
            # Time embedding generation
            start_time = time.time()
            query_embedding = model.encode([test_query])
            doc_embedding = model.encode([test_document])
            embedding_time = time.time() - start_time
            
            # Calculate similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity(query_embedding, doc_embedding)[0][0]
            
            # Get embedding dimensions
            embedding_dim = query_embedding.shape[1]
            
            embedding_results[model_name] = {
                'embedding_time': embedding_time,
                'similarity': similarity,
                'dimensions': embedding_dim,
                'model_size': f"~{embedding_dim * 4 / 1024 / 1024:.1f}MB"  # Rough estimate
            }
            
            print(f"  ⏱️ Embedding time: {embedding_time:.3f} seconds")
            print(f"  📊 Similarity: {similarity:.3f}")
            print(f"  📏 Dimensions: {embedding_dim}")
            
        except Exception as e:
            print(f"  ❌ Error with {model_name}: {e}")
            embedding_results[model_name] = {'error': str(e)}
    
    # Recommend best model
    print(f"\\n🏆 Model Recommendation:")
    if embedding_results:
        # Find best balance of speed and quality
        valid_results = {k: v for k, v in embedding_results.items() if 'error' not in v}
        if valid_results:
            best_model = max(valid_results.keys(), 
                           key=lambda k: valid_results[k]['similarity'] / (valid_results[k]['embedding_time'] + 0.1))
            print(f"Recommended: {best_model}")
            print(f"Reason: Best balance of similarity ({valid_results[best_model]['similarity']:.3f}) and speed ({valid_results[best_model]['embedding_time']:.3f}s)")
    
    return embedding_results

# Compare embedding models
embedding_comparison = compare_embedding_models()
```

##### **Vector Database Optimization Techniques**
```python
# Cell 8: Vector database optimization
def optimize_vector_database():
    \"\"\"Implement vector database optimizations for better performance\"\"\"
    
    print("⚡ Vector Database Optimization")
    print("=" * 40)
    
    # Test current performance
    test_queries = [
        "internet connection problems",
        "mobile plan pricing",
        "bill payment options"
    ]
    
    print("📊 Performance Before Optimization:")
    baseline_times = []
    
    for query in test_queries:
        start_time = time.time()
        results = workshop_rag.intelligent_retrieval(query, top_k=3)
        query_time = time.time() - start_time
        baseline_times.append(query_time)
        print(f"  '{query}': {query_time:.3f}s")
    
    avg_baseline = sum(baseline_times) / len(baseline_times)
    print(f"Average query time: {avg_baseline:.3f}s")
    
    # Implement caching strategy
    query_cache = {}
    
    def cached_retrieval(query: str, top_k: int = 3):
        \"\"\"Retrieval with caching\"\"\"
        cache_key = f"{query.lower().strip()}_{top_k}"
        
        if cache_key in query_cache:
            return query_cache[cache_key]
        
        result = workshop_rag.intelligent_retrieval(query, top_k=top_k)
        query_cache[cache_key] = result
        return result
    
    # Test cached performance
    print("\\n📊 Performance With Caching:")
    cached_times = []
    
    for query in test_queries:
        # First query (cache miss)
        start_time = time.time()
        results = cached_retrieval(query)
        first_time = time.time() - start_time
        
        # Second query (cache hit)
        start_time = time.time()
        results = cached_retrieval(query)
        cached_time = time.time() - start_time
        
        cached_times.append(cached_time)
        print(f"  '{query}': {first_time:.3f}s → {cached_time:.3f}s (cached)")
    
    avg_cached = sum(cached_times) / len(cached_times)
    speedup = avg_baseline / avg_cached if avg_cached > 0 else float('inf')
    
    print(f"\\nAverage cached time: {avg_cached:.3f}s")
    print(f"Speedup: {speedup:.1f}x faster")
    
    # Memory usage estimation
    import sys
    cache_size = sum(sys.getsizeof(str(v)) for v in query_cache.values()) / 1024  # KB
    print(f"Cache memory usage: {cache_size:.1f} KB")
    
    return {
        'baseline_avg': avg_baseline,
        'cached_avg': avg_cached,
        'speedup': speedup,
        'cache_size_kb': cache_size
    }

# Optimize vector database
optimization_results = optimize_vector_database()
```

---

## 🌅 Morning Session Summary and Transition

### Key Achievements:
- ✅ **Environment Setup**: Complete Google Colab configuration with all required dependencies
- ✅ **Knowledge Base Creation**: Comprehensive telecommunications document processing
- ✅ **Vector Database**: Optimized embedding storage and retrieval system
- ✅ **Performance Optimization**: Caching and speed improvements implemented

### Technical Artifacts Created:
- Fully configured workshop environment
- Processed telecommunications knowledge base (8 documents, 50+ chunks)
- Optimized vector database with intelligent retrieval
- Performance benchmarking and caching system

### Preparation for Afternoon:
The morning session establishes a solid technical foundation with a working RAG system. Participants now have:
- Functional vector database with telecommunications content
- Optimized retrieval mechanisms
- Performance monitoring capabilities
- Ready-to-use development environment

**Transition to Afternoon**: With the core RAG infrastructure in place, the afternoon session will focus on building customer-facing applications, implementing advanced features, and collaborative project development.

---

*Continue to Afternoon Session for chatbot development, real-world testing, and group projects...*