# 🕸️ Hyperlink Network & Community Detection on Reddit


> **Advanced text analytics meets graph neural networks**: Extracting community structures and thematic patterns from large-scale Reddit networks using classical algorithms and modern deep learning approaches.

## 🎯 Project Overview

This project analyzes a **2,000-node Reddit hyperlink network** to uncover hidden community structures and thematic patterns within online discussions. By combining traditional graph theory with cutting-edge machine learning techniques, it demonstrates how unstructured text data can be transformed into actionable insights about online community dynamics.

### 🔍 Key Research Questions
- How do hyperlink patterns reveal thematic communities in Reddit discussions?
- Can deep learning embeddings outperform classical community detection methods?
- What trade-offs exist between interpretability and predictive performance in network analysis?

## 📊 Dataset & Network Construction

- **Scale**: 2,000 nodes representing Reddit posts/comments with hyperlink connections
- **Data Source**: Reddit hyperlink network extracted from targeted subreddits
- **Network Type**: Directed graph with weighted edges based on hyperlink frequency
- **Preprocessing**: Text cleaning, link validation, and network topology optimization

## 🛠️ Methodology & Technical Approach

### 1. Classical Graph Analysis
- **Girvan-Newman Community Detection**: Hierarchical community identification through edge betweenness centrality
- **Network Topology Analysis**: Degree distribution, clustering coefficients, and centrality measures

### 2. Node Embedding Techniques
- **Node2Vec Implementation**: Tuned random walk parameters to capture both local and global network structure
- **Hyperparameter Optimization**: Systematic tuning of walk length, number of walks, and embedding dimensions
- **Feature Engineering**: Combined structural and textual features for enhanced representation

### 3. Deep Learning Enhancement
- **Graph Convolutional Networks (GCN)**: Enhanced node embeddings using neighborhood aggregation
- **Architecture Design**: Multi-layer GCN with dropout regularization and residual connections
- **Training Strategy**: Supervised learning with community labels for embedding optimization

### 4. Clustering & Community Detection
- **K-means Clustering**: Applied to both Node2Vec and GCN embeddings for community identification
- **Consensus Clustering**: Combined multiple clustering results for robust community detection

## 📈 Evaluation & Benchmarking

### Performance Metrics
- **Adjusted Rand Index (ARI)**: Measures clustering similarity with ground truth
- **Silhouette Score**: Evaluates cluster cohesion and separation
- **Calinski-Harabasz Index**: Assesses cluster validity through variance ratios
- **Davies-Bouldin Index**: Measures average similarity between clusters



## 🔧 Technologies & Libraries

### Core Technologies
- **Python 3.8+**: Primary programming language
- **NetworkX**: Graph construction and analysis
- **Node2Vec**: Network embedding generation
- **PyTorch**: Deep learning framework for GCN implementation
- **scikit-learn**: Clustering algorithms and evaluation metrics

### Supporting Libraries
```python
pandas>=1.3.0          # Data manipulation
numpy>=1.21.0           # Numerical computing
matplotlib>=3.5.0       # Data visualization  
seaborn>=0.11.0         # Statistical plotting
gensim>=4.1.0          # Word2Vec and Node2Vec
torch-geometric>=2.0.0  # Graph neural network utilities
```



### Project Structure
```
├── data/
│   ├── raw/                    # Raw Reddit network data
│   ├── processed/              # Cleaned and preprocessed networks
│   └── embeddings/             # Generated node embeddings
├── src/
│   ├── data_preprocessing.py   # Data cleaning and network construction
│   ├── community_detection.py  # Girvan-Newman implementation
│   ├── node_embeddings.py     # Node2Vec and GCN implementations
│   ├── clustering_analysis.py # K-means and evaluation metrics
│   └── visualization.py       # Network and community plotting
├── notebooks/
│   └── community_detection_analysis.ipynb
├── results/
│   ├── figures/               # Generated visualizations
│   ├── embeddings/            # Saved embedding models
│   └── evaluation_metrics.csv # Performance comparison results
└── README.md
```

## 📸 Key Visualizations

### Network Structure & Communities
- **Community Network Visualization**: 2D projection showing detected communities with distinct colors
- **Embedding Space Visualization**: t-SNE plots of Node2Vec and GCN embeddings
- **Community Hierarchy**: Dendrogram showing hierarchical community structure

### Performance Comparisons
- **Metric Comparison Charts**: Radar plots comparing all evaluation metrics across methods
- **Stability Analysis**: Performance consistency across different random initializations

## 🧠 Key Insights & Findings

### Technical Discoveries
- **Embedding Quality**: GCN-enhanced embeddings showed superior clustering performance with XX% improvement in ARI score
- **Parameter Sensitivity**: Node2Vec walk length significantly impacts community detection quality
- **Scalability**: Method performance and computational requirements across different network sizes

### Domain Insights  
- **Community Themes**: Identified X distinct thematic communities based on hyperlink patterns
- **Network Topology**: Reddit discussions exhibit small-world properties with clear community boundaries
- **Temporal Patterns**: Community stability and evolution over time in online discussions

## 🎓 Academic & Professional Applications

### Research Contributions
- **Methodological Innovation**: Novel combination of classical and deep learning approaches for network analysis
- **Benchmarking Framework**: Comprehensive evaluation methodology for community detection algorithms
- **Scalability Analysis**: Performance insights for large-scale social network analysis

### Business Applications
- **Content Strategy**: Understanding community themes for targeted content creation
- **User Segmentation**: Data-driven approaches to identify user communities and interests
- **Platform Analytics**: Insights for social media platform optimization and community management

## 🔬 Future Enhancements

- [ ] **Dynamic Community Detection**: Analyze community evolution over time
- [ ] **Multi-layer Networks**: Incorporate multiple relationship types (comments, votes, shares)
- [ ] **Real-time Processing**: Implement streaming algorithms for live community detection
- [ ] **Explainable AI**: Develop interpretability methods for GCN community assignments
- [ ] **Cross-platform Analysis**: Extend methodology to other social media platforms


### Tools & Frameworks
- NetworkX Development Team for graph analysis capabilities
- PyTorch Geometric team for GNN implementations
- Reddit community for providing rich discussion data

## 📄 License & Citation

This project is licensed under the MIT License 





*Built with ❤️ for the data science and network analysis community*





