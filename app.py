import streamlit as st
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration, BartTokenizer, BartForConditionalGeneration
from rouge_score import rouge_scorer
import torch
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Set page config
st.set_page_config(page_title="Text Summarization Comparison", layout="wide")

# Cache model loading
@st.cache_resource
def load_t5_model():
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    return tokenizer, model

@st.cache_resource
def load_bart_model():
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    return tokenizer, model

def summarize_with_t5(text, tokenizer, model, max_length=150, min_length=30):
    # T5 requires "summarize: " prefix
    input_text = "summarize: " + text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    # Ensure min_length doesn't exceed max_length
    actual_min_length = min(min_length, max_length - 10)
    
    start_time = time.time()
    summary_ids = model.generate(
        inputs,
        max_length=max_length,
        min_length=actual_min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    inference_time = time.time() - start_time
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    token_count = len(summary_ids[0])
    return summary, token_count, inference_time

def summarize_with_bart(text, tokenizer, model, max_length=150, min_length=30):
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    
    # Ensure min_length doesn't exceed max_length
    actual_min_length = min(min_length, max_length - 10)
    
    start_time = time.time()
    summary_ids = model.generate(
        inputs,
        max_length=max_length,
        min_length=actual_min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    inference_time = time.time() - start_time
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    token_count = len(summary_ids[0])
    return summary, token_count, inference_time

def calculate_rouge_scores(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores

def calculate_compression_ratio(original_text, summary):
    original_words = len(original_text.split())
    summary_words = len(summary.split())
    return (1 - summary_words / original_words) * 100 if original_words > 0 else 0

def create_rouge_comparison_chart(t5_scores, bart_scores):
    metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    t5_values = [
        t5_scores['rouge1'].fmeasure,
        t5_scores['rouge2'].fmeasure,
        t5_scores['rougeL'].fmeasure
    ]
    bart_values = [
        bart_scores['rouge1'].fmeasure,
        bart_scores['rouge2'].fmeasure,
        bart_scores['rougeL'].fmeasure
    ]
    
    fig = go.Figure(data=[
        go.Bar(name='T5-Small', x=metrics, y=t5_values, marker_color='#FF6B6B'),
        go.Bar(name='BART-Large-CNN', x=metrics, y=bart_values, marker_color='#4ECDC4')
    ])
    
    fig.update_layout(
        title='ROUGE Score Comparison',
        yaxis_title='F1 Score',
        barmode='group',
        height=400,
        template='plotly_white'
    )
    return fig

def create_performance_metrics_chart(t5_time, bart_time, t5_tokens, bart_tokens):
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['T5-Small', 'BART-Large-CNN'],
        y=[t5_time, bart_time],
        name='Inference Time (s)',
        marker_color=['#FF6B6B', '#4ECDC4'],
        text=[f'{t5_time:.3f}s', f'{bart_time:.3f}s'],
        textposition='auto',
    ))
    
    fig.update_layout(
        title='Model Inference Time Comparison',
        yaxis_title='Time (seconds)',
        height=350,
        template='plotly_white',
        showlegend=False
    )
    return fig

# Main app
st.title("📝 Text Summarization Comparison")
st.write("Compare T5-small and BART-large-cnn models with ROUGE score evaluation")

# Text input
st.subheader("Input Text")
input_text = st.text_area(
    "Enter the text you want to summarize:",
    height=200,
    placeholder="Paste your text here..."
)

# Reference summary (optional)
st.subheader("Reference Summary (Optional)")
reference_summary = st.text_area(
    "Enter a reference summary for ROUGE score calculation (optional):",
    height=100,
    placeholder="If you have a gold standard summary, paste it here..."
)

# Summary length slider
max_length = st.slider("Maximum summary length (in tokens):", min_value=50, max_value=300, value=150, step=10)
min_length = st.slider("Minimum summary length (in tokens):", min_value=20, max_value=100, value=30, step=5)

# Summarize button
if st.button("Generate Summaries", type="primary"):
    if not input_text.strip():
        st.error("Please enter some text to summarize!")
    else:
        with st.spinner("Loading models and generating summaries..."):
            # Load models
            t5_tokenizer, t5_model = load_t5_model()
            bart_tokenizer, bart_model = load_bart_model()
            
            # Generate summaries
            t5_summary, t5_tokens, t5_time = summarize_with_t5(input_text, t5_tokenizer, t5_model, max_length, min_length)
            bart_summary, bart_tokens, bart_time = summarize_with_bart(input_text, bart_tokenizer, bart_model, max_length, min_length)
            
            # Calculate compression ratios
            t5_compression = calculate_compression_ratio(input_text, t5_summary)
            bart_compression = calculate_compression_ratio(input_text, bart_summary)
        
        # Display results
        st.success("Summaries generated successfully!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🤖 T5-Small Summary")
            st.write(t5_summary)
            st.caption(f"📊 {len(t5_summary.split())} words | {t5_tokens} tokens | ⚡ {t5_time:.3f}s | 📉 {t5_compression:.1f}% compression")
        
        with col2:
            st.subheader("🤖 BART-Large-CNN Summary")
            st.write(bart_summary)
            st.caption(f"📊 {len(bart_summary.split())} words | {bart_tokens} tokens | ⚡ {bart_time:.3f}s | 📉 {bart_compression:.1f}% compression")
        
        # Performance metrics visualization
        st.subheader("⚡ Performance Metrics")
        perf_chart = create_performance_metrics_chart(t5_time, bart_time, t5_tokens, bart_tokens)
        st.plotly_chart(perf_chart, use_container_width=True, key="perf_chart")
        
        # Detailed metrics table
        st.subheader("📈 Detailed Comparison")
        
        # Use container to prevent flickering
        metrics_df = pd.DataFrame({
            'Metric': ['Words', 'Tokens', 'Inference Time (s)', 'Compression Ratio (%)', 'Speed (words/sec)'],
            'T5-Small': [
                len(t5_summary.split()),
                t5_tokens,
                f'{t5_time:.3f}',
                f'{t5_compression:.1f}',
                f'{len(t5_summary.split())/t5_time:.1f}'
            ],
            'BART-Large-CNN': [
                len(bart_summary.split()),
                bart_tokens,
                f'{bart_time:.3f}',
                f'{bart_compression:.1f}',
                f'{len(bart_summary.split())/bart_time:.1f}'
            ]
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        # Calculate ROUGE scores if reference is provided
        if reference_summary.strip():
            st.subheader("📊 ROUGE Score Analysis")
            st.write("Comparing generated summaries against the reference summary:")
            
            t5_scores = calculate_rouge_scores(reference_summary, t5_summary)
            bart_scores = calculate_rouge_scores(reference_summary, bart_summary)
            
            # Visual comparison
            rouge_chart = create_rouge_comparison_chart(t5_scores, bart_scores)
            st.plotly_chart(rouge_chart, use_container_width=True, key="rouge_chart")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**T5-Small ROUGE Scores:**")
                st.metric("ROUGE-1", f"{t5_scores['rouge1'].fmeasure:.4f}", 
                         delta=f"{(t5_scores['rouge1'].fmeasure - bart_scores['rouge1'].fmeasure):.4f}")
                st.metric("ROUGE-2", f"{t5_scores['rouge2'].fmeasure:.4f}",
                         delta=f"{(t5_scores['rouge2'].fmeasure - bart_scores['rouge2'].fmeasure):.4f}")
                st.metric("ROUGE-L", f"{t5_scores['rougeL'].fmeasure:.4f}",
                         delta=f"{(t5_scores['rougeL'].fmeasure - bart_scores['rougeL'].fmeasure):.4f}")
            
            with col2:
                st.write("**BART-Large-CNN ROUGE Scores:**")
                st.metric("ROUGE-1", f"{bart_scores['rouge1'].fmeasure:.4f}",
                         delta=f"{(bart_scores['rouge1'].fmeasure - t5_scores['rouge1'].fmeasure):.4f}")
                st.metric("ROUGE-2", f"{bart_scores['rouge2'].fmeasure:.4f}",
                         delta=f"{(bart_scores['rouge2'].fmeasure - t5_scores['rouge2'].fmeasure):.4f}")
                st.metric("ROUGE-L", f"{bart_scores['rougeL'].fmeasure:.4f}",
                         delta=f"{(bart_scores['rougeL'].fmeasure - t5_scores['rougeL'].fmeasure):.4f}")
            
            # Determine winner
            st.subheader("🏆 Model Recommendation")
            avg_t5 = (t5_scores['rouge1'].fmeasure + t5_scores['rouge2'].fmeasure + t5_scores['rougeL'].fmeasure) / 3
            avg_bart = (bart_scores['rouge1'].fmeasure + bart_scores['rouge2'].fmeasure + bart_scores['rougeL'].fmeasure) / 3
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("T5-Small Avg ROUGE", f"{avg_t5:.4f}")
            with col2:
                st.metric("BART-Large-CNN Avg ROUGE", f"{avg_bart:.4f}")
            
            if avg_t5 > avg_bart:
                st.success(f"✅ T5-Small performs better with an average ROUGE score of {avg_t5:.4f} (Quality winner)")
            elif avg_bart > avg_t5:
                st.success(f"✅ BART-Large-CNN performs better with an average ROUGE score of {avg_bart:.4f} (Quality winner)")
            else:
                st.info("Both models perform equally well in terms of ROUGE scores!")
            
            if t5_time < bart_time:
                st.info(f"⚡ T5-Small is {bart_time/t5_time:.2f}x faster (Speed winner)")
            else:
                st.info(f"⚡ BART-Large-CNN is {t5_time/bart_time:.2f}x faster (Speed winner)")
        else:
            st.info("💡 Tip: Add a reference summary to calculate ROUGE scores and compare model performance!")

# Sidebar with info
with st.sidebar:
    st.header("About This Project")
    st.write("""
    A comprehensive NLP application comparing state-of-the-art 
    text summarization models with quantitative evaluation metrics.
    """)
    
    st.header("Models")
    st.write("""
    **T5-Small:**
    - Text-to-Text Transfer Transformer
    - 60M parameters
    - Faster inference (~2-3x)
    - Lower memory footprint
    
    **BART-Large-CNN:**
    - Bidirectional Auto-Regressive Transformer
    - 406M parameters
    - Fine-tuned on CNN/DailyMail
    - Higher quality summaries
    """)
    
    st.header("Evaluation Metrics")
    st.write("""
    **ROUGE Scores:**
    - ROUGE-1: Unigram overlap
    - ROUGE-2: Bigram overlap
    - ROUGE-L: Longest common subsequence
    
    **Performance Metrics:**
    - Inference time
    - Compression ratio
    - Throughput (words/sec)
    """)
    
    st.header("Technical Stack")
    st.code("""
    • Streamlit
    • HuggingFace Transformers
    • PyTorch
    • ROUGE Score
    • Plotly
    """, language="text")
    
    st.header("Sample Text")
    if st.button("Load Sample"):
        st.session_state.sample_loaded = True
    
    if st.session_state.get('sample_loaded'):
        st.code("""
The Amazon rainforest is the world's largest tropical rainforest, covering over 5.5 million square kilometers. 
It spans across nine countries in South America, with the majority located in Brazil. The rainforest is home to 
an estimated 390 billion individual trees divided into 16,000 species. It plays a crucial role in regulating 
the global climate and is often referred to as the "lungs of the Earth" because it produces about 20% of the 
world's oxygen. The Amazon is also incredibly biodiverse, hosting millions of species of insects, plants, birds, 
and other forms of life, many of which are still undiscovered. However, deforestation poses a significant threat 
to this vital ecosystem, with large areas being cleared for agriculture and logging each year.
        """)
