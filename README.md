# 📝 Text Summarization Comparison App

🚀 **[Live Demo](https://llmeval-dpyznpyt4rkk6mvcpywmyq.streamlit.app/)** - Try it now!

An advanced NLP application that performs comparative analysis of state-of-the-art text summarization models using HuggingFace Transformers, with comprehensive evaluation metrics and interactive visualizations.

## 🎯 Project Overview

This project demonstrates practical implementation of Natural Language Processing techniques for automatic text summarization, featuring:
- Multi-model comparison framework
- Quantitative evaluation using ROUGE metrics
- Performance benchmarking and analysis
- Interactive web interface using Streamlit

## ✨ Key Features

### Model Comparison
- **T5-Small**: Text-to-Text Transfer Transformer (60M parameters)
  - Faster inference time
  - Lower memory footprint
  - Suitable for real-time applications
  
- **BART-Large-CNN**: Bidirectional Auto-Regressive Transformer (406M parameters)
  - Fine-tuned on CNN/DailyMail dataset
  - Higher quality summaries
  - Better for accuracy-critical tasks

### Evaluation Metrics
- **ROUGE Scores**: Industry-standard metrics for summarization quality
  - ROUGE-1: Unigram overlap
  - ROUGE-2: Bigram overlap
  - ROUGE-L: Longest common subsequence
  
- **Performance Metrics**:
  - Inference time measurement
  - Compression ratio calculation
  - Throughput analysis (words/second)

### Interactive Visualizations
- Side-by-side summary comparison
- ROUGE score bar charts
- Performance metrics visualization
- Detailed comparison tables
- Real-time metric deltas

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **ML Framework**: PyTorch, HuggingFace Transformers
- **Evaluation**: ROUGE Score
- **Visualization**: Plotly, Pandas
- **Language**: Python 3.8+

## 📦 Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd text-summarization-app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

The app will launch in your browser at `http://localhost:8501`

## 🚀 Usage Guide

1. **Input Text**: Paste or type the text you want to summarize
2. **Reference Summary** (Optional): Add a gold-standard summary for ROUGE evaluation
3. **Adjust Parameters**: 
   - Maximum summary length (tokens)
   - Minimum summary length (tokens)
4. **Generate**: Click "Generate Summaries" to process
5. **Analyze Results**:
   - Compare summaries side-by-side
   - Review performance metrics
   - Examine ROUGE scores (if reference provided)
   - View interactive visualizations

## 📊 Output Metrics

### Quality Metrics
- ROUGE-1, ROUGE-2, ROUGE-L F1 scores
- Precision and recall values
- Average ROUGE score

### Performance Metrics
- Inference time (seconds)
- Compression ratio (%)
- Processing speed (words/second)
- Token count vs word count

## 🎓 Learning Outcomes

This project demonstrates:
- Implementation of transformer-based NLP models
- Model comparison and benchmarking techniques
- Evaluation metrics for text generation tasks
- Building interactive ML applications
- Data visualization for model analysis
- Production-ready code structure

## 📈 Use Cases

- Document summarization
- News article condensation
- Research paper abstracts
- Meeting notes summarization
- Content curation
- Model selection for production deployment

## ⚠️ Important Notes

- First run downloads models (~1.5GB total)
- Requires stable internet connection for initial setup
- GPU recommended for faster inference (optional)
- Models are cached after first download

## 🔬 Future Enhancements

- [ ] Add more models (Pegasus, LED, etc.)
- [ ] Batch processing capability
- [ ] Export summaries to PDF/DOCX
- [ ] Fine-tuning interface
- [ ] Multi-language support
- [ ] API endpoint creation

## 📝 Project Structure

```
text-summarization-app/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── .gitignore           # Git ignore file
```

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

Created as a demonstration of NLP and ML engineering skills for data science applications.

## 🙏 Acknowledgments

- HuggingFace for transformer models
- Streamlit for the web framework
- Google Research for T5
- Facebook AI for BART
