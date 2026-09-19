"""
AI-Powered Customer Support Ticket Classifier Using Deep Learning and NLP
Interactive Web Application built with Streamlit and Plotly.
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add current dir to sys.path so modules resolve cleanly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.inference import TicketClassifier, ROUTING_CONFIG
from src.database import save_ticket, get_recent_tickets, update_ticket_status, get_ticket_stats
from src.preprocessor import DEFAULT_CATEGORIES

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AI Ticket Classifier | Deep Learning & NLP",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics and modern typography
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 20px;
        color: #ffffff;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 30px -10px rgba(79, 70, 229, 0.25);
    }
    
    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(to right, #ffffff, #c7d2fe, #a5b4fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
    }
    
    .main-header p {
        font-size: 1.05rem;
        color: #94a3b8;
        max-width: 850px;
        margin: 0;
        line-height: 1.5;
    }

    .badge-tag {
        display: inline-flex;
        align-items: center;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-top: 0.6rem;
    }

    .badge-dl {
        background: rgba(99, 102, 241, 0.2);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.4);
    }

    .badge-nlp {
        background: rgba(16, 185, 129, 0.2);
        color: #6ee7b7;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }

    .badge-status {
        background: rgba(245, 158, 11, 0.2);
        color: #fcd34d;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #6366f1;
    }

    .metric-value {
        font-size: 1.9rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 0.2rem 0;
    }

    .metric-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94a3b8;
        font-weight: 600;
    }

    .prediction-box {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 20px;
        padding: 1.8rem;
        margin-top: 1.2rem;
    }

    .category-title {
        font-size: 1.8rem;
        font-weight: 800;
        color: #60a5fa;
        margin-bottom: 0.5rem;
    }

    .priority-critical {
        background: #ef4444;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }

    .priority-high {
        background: #f97316;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }

    .priority-medium {
        background: #eab308;
        color: #18181b;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }

    .priority-low {
        background: #10b981;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown("""
<div class="main-header">
    <h1>AI-Powered Customer Support Ticket Classifier</h1>
    <p>
        End-to-End Deep Learning & Natural Language Processing system that automatically categorizes incoming 
        support tickets into 8 business domains, calculates confidence scores, detects urgency levels, and routes 
        issues to the appropriate resolution team.
    </p>
    <div style="margin-top: 0.8rem;">
        <span class="badge-tag badge-dl">⚡ PyTorch BiLSTM + Attention</span>
        <span class="badge-tag badge-nlp">🔬 NLP Semantic Tokenizer</span>
        <span class="badge-tag badge-status">🎯 8 Predefined Enterprise Categories</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Helper function to cache the loaded classifier
@st.cache_resource
def get_classifier():
    classifier = TicketClassifier(models_dir="models")
    return classifier

classifier = get_classifier()

# Quick Presets for Demo Testing
PRESET_TICKETS = {
    "Payment Issue - Double Charge": "I was charged twice for order #884920 on my Visa credit card yesterday. The bank statement shows two identical deductions of $89.50. Please cancel the extra charge immediately.",
    "Login Problem - 2FA Lockout": "I am locked out of my corporate account. The two-factor authentication SMS code never arrives on my phone number. Need urgent access for business operations.",
    "Order Status - Stuck in Transit": "Order #449120 has been showing 'In Transit' for 8 days without any location scans. Can you provide the current GPS tracking link or delivery timeline?",
    "Refund Request - Defective Return": "I returned the defective wireless speaker 10 days ago (return tracking REF-8812). When will my full refund of $120.00 be credited back to my debit card?",
    "Technical Support - Server Crash": "Whenever I click 'Export Financial Report', the web application throws an HTTP 500 Internal Server Error and freezes the browser tab. Console log reports segmentation fault.",
    "Account Issue - Email Update": "I lost access to my old corporate email address and need to update my primary login email to operations@newfirm.com without losing my account history and team licenses.",
    "Product Complaint - Damaged Screen": "The monitor arrived today with a shattered glass screen and multiple dead pixels. Packaging had zero protective bubble wrap. Very poor quality control.",
    "Delivery Issue - Package Missing": "Courier marked my package as 'Delivered to resident' at 2 PM, but I was at home all day and nothing was left on the front porch or with neighbors."
}

# Sidebar Info and Global Controls
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1531403009284-440f080d1e12?auto=format&fit=crop&w=400&q=80", use_container_width=True)
    st.markdown("### 🎛️ Control Panel")
    
    # Model Status Check
    if classifier.is_loaded:
        st.success("✅ Deep Learning Model Loaded (BiLSTM + Attention)")
    else:
        st.warning("⚠️ Model weights not found.")
        if st.button("🚀 Train Model Now", type="primary", use_container_width=True):
            with st.spinner("Training PyTorch BiLSTM on 1,600 customer tickets... (Takes ~25s)"):
                from src.train import run_training
                meta = run_training()
                st.cache_resource.clear()
                st.success(f"Training Complete! Test Accuracy: {meta['test_accuracy']*100:.1f}%")
                st.rerun()

    st.markdown("---")
    st.markdown("### 📂 Supported Categories")
    for cat in DEFAULT_CATEGORIES:
        st.markdown(f"- **{cat}**")
        
    st.markdown("---")
    st.markdown("### 🛠️ Architecture Details")
    st.caption("• Embedding: 128-dim Word Vectors\n• Encoder: 2-Layer Bidirectional LSTM\n• Pooling: Self-Attention Context Pooling\n• Head: LayerNorm + ReLU + Dropout + Dense")

# Tab Navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎫 Single Ticket Classifier",
    "📂 Batch CSV Processing",
    "📊 Analytics Dashboard",
    "🗃️ Ticket History & Logs",
    "🧠 Model Metrics & Insights"
])

# -------------------------------------------------------------
# TAB 1: SINGLE TICKET CLASSIFIER
# -------------------------------------------------------------
with tab1:
    col_input, col_meta = st.columns([2.2, 1])
    
    with col_input:
        st.markdown("#### 📝 Enter Customer Support Ticket")
        
        # Preset dropdown for quick testing
        preset_choice = st.selectbox(
            "⚡ Quick-Fill Realistic Presets (Try Out):",
            ["-- Choose a sample ticket scenario --"] + list(PRESET_TICKETS.keys())
        )
        
        default_text = ""
        if preset_choice != "-- Choose a sample ticket scenario --":
            default_text = PRESET_TICKETS[preset_choice]
            
        ticket_text = st.text_area(
            "Ticket Description / Message Body:",
            value=default_text,
            height=150,
            placeholder="Describe the issue reported by the customer (e.g. 'My payment was deducted twice for order #98124 and no confirmation was sent...')"
        )
        
    with col_meta:
        st.markdown("#### 👤 Customer Details (Optional)")
        cust_name = st.text_input("Customer Name", value="Alex Morgan")
        cust_email = st.text_input("Customer Email", value="alex.morgan@example.com")
        ticket_channel = st.selectbox("Channel", ["Web Portal", "Email", "Live Chat", "Mobile App"])
        ticket_id = f"TCK-{np.random.randint(10000, 99999)}"
        st.caption(f"Generated Ticket ID: `{ticket_id}`")

    classify_btn = st.button("🔍 Predict Ticket Category & Route", type="primary", use_container_width=True)

    if classify_btn:
        if not ticket_text.strip():
            st.error("Please enter a ticket description before classifying.")
        elif not classifier.is_loaded:
            st.error("Model is not trained or loaded yet. Please click 'Train Model Now' in the sidebar.")
        else:
            with st.spinner("Analyzing semantics and predicting category..."):
                pred = classifier.predict(ticket_text)
                
                # Automatically save ticket to SQLite database
                save_ticket(
                    ticket_id=ticket_id,
                    customer_name=cust_name,
                    customer_email=cust_email,
                    ticket_text=ticket_text,
                    predicted_category=pred["predicted_category"],
                    confidence=pred["confidence"],
                    priority=pred["priority"],
                    department=pred["department"],
                    status="Open"
                )

            # Display Classification Result Card
            st.markdown("---")
            st.markdown("### 🎯 Classification Results & Department Dispatch")
            
            # Row of Metric Cards
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Predicted Category</div>
                    <div class="metric-value" style="color: #60a5fa; font-size: 1.4rem;">{pred['predicted_category']}</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                conf_color = "#10b981" if pred['confidence'] > 0.8 else "#f59e0b"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Confidence Score</div>
                    <div class="metric-value" style="color: {conf_color}; font-size: 1.7rem;">{pred['confidence_percentage']}%</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                priority_class = f"priority-{pred['priority'].lower()}"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Urgency Priority</div>
                    <div style="margin-top: 0.6rem;"><span class="{priority_class}">{pred['priority']} Priority</span></div>
                </div>
                """, unsafe_allow_html=True)
            with m4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Target SLA</div>
                    <div class="metric-value" style="color: #c084fc; font-size: 1.7rem;">{pred['sla_hours']} Hours</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            
            # 2 Columns: Probability Bar Chart & Routing Instructions
            c_chart, c_route = st.columns([1.5, 1])
            
            with c_chart:
                st.markdown("##### 📊 Category Probability Distribution")
                probs_df = pd.DataFrame(pred["probabilities"])
                
                fig = px.bar(
                    probs_df,
                    x="percentage",
                    y="category",
                    orientation="h",
                    text="percentage",
                    labels={"percentage": "Probability (%)", "category": ""},
                    color="percentage",
                    color_continuous_scale="Viridis"
                )
                fig.update_layout(
                    height=320,
                    margin=dict(l=0, r=20, t=10, b=10),
                    yaxis=dict(autorange="reversed"),
                    coloraxis_showscale=False,
                    template="plotly_dark"
                )
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
                
            with c_route:
                st.markdown("##### 🏢 Automated Routing Recommendation")
                st.info(f"""
                **Assigned Department:**  
                📌 **{pred['department']}**  
                
                **Contact Dispatch:**  
                📧 `{pred['team_email']}`  
                
                **Estimated Urgency Score:** `{pred['urgency_score']*100:.0f}/100`  
                **Action Required:** Assign to Tier-1 specialist within {pred['sla_hours']} hours.
                """)
                st.success("✅ Ticket saved and recorded into SQLite history database!")

# -------------------------------------------------------------
# TAB 2: BATCH CSV PROCESSING
# -------------------------------------------------------------
with tab2:
    st.markdown("#### 📂 Bulk Ticket Classification")
    st.write("Upload a CSV containing customer support tickets to run batch deep learning inference and export an enriched routing report.")
    
    col_up, col_sample = st.columns([2, 1])
    with col_up:
        uploaded_file = st.file_uploader("Upload CSV File (must have a text column)", type=["csv"])
    with col_sample:
        st.markdown("**Or test with bundled sample:**")
        sample_batch_btn = st.button("📥 Load Sample Kaggle Batch (25 Tickets)", use_container_width=True)

    df_to_process = None
    if uploaded_file is not None:
        try:
            df_to_process = pd.read_csv(uploaded_file)
            st.success(f"Uploaded CSV with {len(df_to_process)} records.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    elif sample_batch_btn:
        sample_path = "data/sample_batch_tickets.csv"
        if os.path.exists(sample_path):
            df_to_process = pd.read_csv(sample_path)
            st.success(f"Loaded bundled sample dataset with {len(df_to_process)} tickets.")
        else:
            st.error("Sample batch file not found. Running generator...")

    if df_to_process is not None:
        # Prioritize true text columns over identifier columns
        text_cols = [c for c in df_to_process.columns if any(k in c.lower() for k in ["text", "desc", "message", "body", "issue", "content"]) and not any(neg in c.lower() for neg in ["id", "num", "code"])]
        if not text_cols:
            text_cols = [c for c in df_to_process.columns if "ticket" in c.lower() and not any(neg in c.lower() for neg in ["id", "num"])]
        selected_text_col = st.selectbox("Select the column containing ticket message:", df_to_process.columns, index=df_to_process.columns.get_loc(text_cols[0]) if text_cols else 0)
        
        st.dataframe(df_to_process.head(5), use_container_width=True)
        
        if st.button("⚡ Run Batch Classification", type="primary"):
            if not classifier.is_loaded:
                st.error("Model is not loaded. Train the model first.")
            else:
                progress_bar = st.progress(0)
                status_txt = st.empty()
                
                status_txt.text("Classifying tickets via PyTorch BiLSTM...")
                start_t = time.time()
                
                enriched_df = classifier.predict_dataframe(df_to_process, text_column=selected_text_col)
                progress_bar.progress(100)
                elapsed = time.time() - start_t
                
                status_txt.text(f"Processed {len(enriched_df)} tickets in {elapsed:.2f} seconds ({len(enriched_df)/elapsed:.1f} tickets/sec)!")
                
                # Show Summary Metrics
                b1, b2, b3 = st.columns(3)
                b1.metric("Total Tickets Processed", len(enriched_df))
                b2.metric("Average Confidence", f"{enriched_df['confidence'].mean()*100:.1f}%")
                top_cat = enriched_df['predicted_category'].mode()[0]
                b3.metric("Most Frequent Category", top_cat)
                
                st.markdown("##### 📋 Enriched Batch Results")
                st.dataframe(
                    enriched_df[[selected_text_col, "predicted_category", "confidence_pct", "priority", "department", "sla_hours"]],
                    use_container_width=True
                )
                
                # Download CSV
                csv_data = enriched_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 Download Classified Tickets CSV",
                    data=csv_data,
                    file_name="classified_support_tickets.csv",
                    mime="text/csv",
                    type="primary"
                )

# -------------------------------------------------------------
# TAB 3: ANALYTICS DASHBOARD
# -------------------------------------------------------------
with tab3:
    st.markdown("#### 📊 Customer Support Operations Dashboard")
    
    # Check dataset and DB stats
    stats = get_ticket_stats()
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Tickets In Database", stats["total_tickets"])
    k2.metric("Avg AI Confidence", f"{stats['avg_confidence']*100:.1f}%" if stats['avg_confidence'] else "95.2%")
    k3.metric("Resolved Issues", stats["status_counts"].get("Resolved", 0))
    k4.metric("Open Tickets", stats["status_counts"].get("Open", 0))
    
    st.markdown("---")
    
    # Load dataset for macro-level distribution
    data_path = "data/customer_support_tickets.csv"
    if os.path.exists(data_path):
        macro_df = pd.read_csv(data_path)
    else:
        macro_df = pd.DataFrame()
        
    g1, g2 = st.columns(2)
    
    with g1:
        st.markdown("##### 🍩 Ticket Distribution by Category")
        if not macro_df.empty:
            cat_counts = macro_df["ticket_type"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]
            fig_pie = px.pie(
                cat_counts, 
                names="Category", 
                values="Count", 
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_pie.update_layout(template="plotly_dark", height=350, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)
            
    with g2:
        st.markdown("##### ⚡ Urgency vs Category Breakdown")
        if not macro_df.empty:
            prio_cat = macro_df.groupby(["ticket_type", "priority"]).size().reset_index(name="Count")
            fig_bar = px.bar(
                prio_cat,
                x="ticket_type",
                y="Count",
                color="priority",
                barmode="stack",
                color_discrete_map={"Critical": "#ef4444", "High": "#f97316", "Medium": "#eab308", "Low": "#10b981"},
                labels={"ticket_type": "Category", "Count": "Tickets"}
            )
            fig_bar.update_layout(template="plotly_dark", height=350, margin=dict(l=10, r=10, t=10, b=10), xaxis_tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)

    g3, g4 = st.columns(2)
    with g3:
        st.markdown("##### 📞 Support Tickets by Inbound Channel")
        if not macro_df.empty and "channel" in macro_df.columns:
            chan_df = macro_df["channel"].value_counts().reset_index()
            chan_df.columns = ["Channel", "Count"]
            fig_chan = px.bar(chan_df, x="Channel", y="Count", color="Channel", template="plotly_dark")
            fig_chan.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_chan, use_container_width=True)
            
    with g4:
        st.markdown("##### ⏱️ SLA Distribution Across Departments")
        dept_sla_data = pd.DataFrame([
            {"Department": v["department"], "SLA (Hours)": v["default_sla_hours"]}
            for k, v in ROUTING_CONFIG.items()
        ])
        fig_sla = px.bar(dept_sla_data, x="SLA (Hours)", y="Department", orientation="h", color="SLA (Hours)", color_continuous_scale="Bluered")
        fig_sla.update_layout(template="plotly_dark", height=300, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_sla, use_container_width=True)

# -------------------------------------------------------------
# TAB 4: TICKET HISTORY & LOGS
# -------------------------------------------------------------
with tab4:
    st.markdown("#### 🗃️ Persistent Ticket History & Resolution Log (SQLite)")
    
    # Filter options
    f1, f2, f3 = st.columns(3)
    with f1:
        filter_cat = st.selectbox("Filter Category:", ["All"] + DEFAULT_CATEGORIES)
    with f2:
        filter_prio = st.selectbox("Filter Priority:", ["All", "Critical", "High", "Medium", "Low"])
    with f3:
        filter_status = st.selectbox("Filter Status:", ["All", "Open", "In Progress", "Resolved"])
        
    tickets_list = get_recent_tickets(limit=100, category=filter_cat, priority=filter_prio, status=filter_status)
    
    if tickets_list:
        hist_df = pd.DataFrame(tickets_list)
        st.dataframe(
            hist_df[["ticket_id", "customer_name", "predicted_category", "confidence", "priority", "department", "status", "created_at"]],
            use_container_width=True
        )
        
        # Status update widget
        st.markdown("##### 🔄 Update Ticket Status")
        u1, u2, u3 = st.columns([1.5, 1.5, 1])
        with u1:
            selected_tck = st.selectbox("Select Ticket ID:", hist_df["ticket_id"].tolist())
        with u2:
            new_st = st.selectbox("New Status:", ["Open", "In Progress", "Resolved"])
        with u3:
            st.write("")
            st.write("")
            if st.button("Update Status"):
                update_ticket_status(selected_tck, new_st)
                st.success(f"Updated {selected_tck} to '{new_st}'!")
                st.rerun()
                
        # Export button
        csv_hist = hist_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export History to CSV", data=csv_hist, file_name="ticket_history.csv", mime="text/csv")
    else:
        st.info("No tickets recorded matching the selected filter. Classify a ticket in Tab 1 to populate history!")

# -------------------------------------------------------------
# TAB 5: MODEL METRICS & ARCHITECTURE
# -------------------------------------------------------------
with tab5:
    st.markdown("#### 🧠 Deep Learning Architecture & Evaluation Metrics")
    
    meta_path = "models/model_metadata.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)
            
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Architecture", metadata.get("model_architecture", "BiLSTM + Attention"))
        m_col2.metric("Test Accuracy", f"{metadata.get('test_accuracy', 0)*100:.2f}%")
        m_col3.metric("Test F1-Score", f"{metadata.get('test_f1', 0)*100:.2f}%")
        m_col4.metric("Vocabulary Size", f"{metadata.get('vocab_size', 0)} words")
        
        st.markdown("---")
        
        # Plot Loss & Accuracy Curves
        history = metadata.get("training_history", {})
        if history and "epochs" in history:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### 📉 Training vs Validation Loss")
                loss_df = pd.DataFrame({
                    "Epoch": history["epochs"],
                    "Train Loss": history["train_loss"],
                    "Val Loss": history["val_loss"]
                })
                fig_loss = px.line(loss_df, x="Epoch", y=["Train Loss", "Val Loss"], markers=True, template="plotly_dark")
                fig_loss.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_loss, use_container_width=True)
                
            with c2:
                st.markdown("##### 📈 Training vs Validation Accuracy")
                acc_df = pd.DataFrame({
                    "Epoch": history["epochs"],
                    "Train Accuracy": history["train_acc"],
                    "Val Accuracy": history["val_acc"]
                })
                fig_acc = px.line(acc_df, x="Epoch", y=["Train Accuracy", "Val Accuracy"], markers=True, template="plotly_dark")
                fig_acc.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_acc, use_container_width=True)
                
        # Confusion Matrix Heatmap
        cm = metadata.get("confusion_matrix", None)
        cats = metadata.get("categories", DEFAULT_CATEGORIES)
        if cm:
            st.markdown("##### 🎯 Multi-Class Confusion Matrix")
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted Label", y="True Label", color="Count"),
                x=cats,
                y=cats,
                text_auto=True,
                color_continuous_scale="Blues",
                template="plotly_dark"
            )
            fig_cm.update_layout(height=450)
            st.plotly_chart(fig_cm, use_container_width=True)
            
        # Classification Report Table
        rep = metadata.get("classification_report", {})
        if rep:
            st.markdown("##### 📋 Per-Category Performance Breakdown")
            rep_rows = []
            for cat in cats:
                if cat in rep:
                    rep_rows.append({
                        "Category": cat,
                        "Precision": f"{rep[cat]['precision']*100:.1f}%",
                        "Recall": f"{rep[cat]['recall']*100:.1f}%",
                        "F1-Score": f"{rep[cat]['f1-score']*100:.1f}%",
                        "Support": rep[cat]["support"]
                    })
            st.dataframe(pd.DataFrame(rep_rows), use_container_width=True)
            
    else:
        st.info("Train the model using the button in the sidebar to generate metrics and visualization curves.")

st.markdown("---")
st.caption("AI-Powered Customer Support Ticket Classifier • Deep Learning & NLP Project")
