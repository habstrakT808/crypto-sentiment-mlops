# File: src/monitoring/dashboard/components/model_comparison.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def show_model_comparison():
    """Display model comparison analysis"""
    
    st.markdown("## 🔬 Model Comparison")
    st.markdown("Compare performance across different sentiment analysis models")
    
    # Model data (from your training results)
    models_data = {
        'Model': ['Baseline\n(LogReg)', 'LSTM', 'BERT', 'FinBERT', 'LightGBM', 'Ensemble'],
        'Accuracy': [0.878, 0.923, 0.538, 0.615, 0.846, 0.851],
        'F1_Weighted': [0.842, 0.923, 0.486, 0.590, 0.834, 0.847],
        'F1_Macro': [0.422, 0.423, 0.428, 0.511, 0.549, 0.523],
        'Training_Time': [1, 300, 1800, 1200, 60, 400],  # seconds
        'Inference_Time': [0.01, 0.15, 0.50, 0.40, 0.05, 0.20],  # seconds
        'Model_Size': [0.1, 50, 440, 440, 5, 55],  # MB
        'Status': ['✅ Realistic', '✅ Realistic', '⚠️ Needs Data', '⚠️ Needs Data', '✅ Production', '✅ Production']
    }
    
    df = pd.DataFrame(models_data)
    
    # Display comparison table
    st.markdown("### 📊 Performance Comparison")
    
    # Highlight best values
    def highlight_best(s):
        if s.name in ['Accuracy', 'F1_Weighted', 'F1_Macro']:
            is_max = s == s.max()
            return ['background-color: #00ff8822' if v else '' for v in is_max]
        elif s.name in ['Training_Time', 'Inference_Time', 'Model_Size']:
            is_min = s == s.min()
            return ['background-color: #00ff8822' if v else '' for v in is_min]
        return ['' for _ in s]
    
    st.dataframe(
        df.style.apply(highlight_best),
        use_container_width=True,
        height=300
    )
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy comparison
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Bar(
            x=df['Model'],
            y=df['Accuracy'],
            text=[f"{acc:.1%}" for acc in df['Accuracy']],
            textposition='outside',
            marker_color=px.colors.sequential.Viridis,
            name='Accuracy'
        ))
        
        fig_acc.update_layout(
            title="🎯 Model Accuracy Comparison",
            xaxis_title="Model",
            yaxis_title="Accuracy",
            yaxis=dict(range=[0, 1]),
            height=400
        )
        
        st.plotly_chart(fig_acc, use_container_width=True)
    
    with col2:
        # F1 Score comparison
        fig_f1 = go.Figure()
        
        fig_f1.add_trace(go.Bar(
            x=df['Model'],
            y=df['F1_Weighted'],
            name='F1 Weighted',
            marker_color='#667eea'
        ))
        
        fig_f1.add_trace(go.Bar(
            x=df['Model'],
            y=df['F1_Macro'],
            name='F1 Macro',
            marker_color='#764ba2'
        ))
        
        fig_f1.update_layout(
            title="📊 F1 Score Comparison",
            xaxis_title="Model",
            yaxis_title="F1 Score",
            yaxis=dict(range=[0, 1]),
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig_f1, use_container_width=True)
    
    # Performance vs Efficiency
    st.markdown("### ⚡ Performance vs Efficiency")
    
    fig_scatter = go.Figure()
    
    fig_scatter.add_trace(go.Scatter(
        x=df['Inference_Time'],
        y=df['Accuracy'],
        mode='markers+text',
        text=df['Model'],
        textposition="top center",
        marker=dict(
            size=df['Model_Size'],
            sizemode='diameter',
            sizeref=max(df['Model_Size'])/50,
            color=df['Accuracy'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Accuracy"),
            line=dict(width=2, color='white')
        ),
        hovertemplate='<b>%{text}</b><br>' +
                      'Accuracy: %{y:.1%}<br>' +
                      'Inference Time: %{x:.2f}s<br>' +
                      '<extra></extra>'
    ))
    
    fig_scatter.update_layout(
        title="🎯 Accuracy vs Inference Speed (bubble size = model size)",
        xaxis_title="Inference Time (seconds)",
        yaxis_title="Accuracy",
        height=500,
        hovermode='closest'
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Recommendations
    st.markdown("### 💡 Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #00ff8822 0%, #00ff8844 100%); padding: 1rem; border-radius: 10px;">
            <h4>🏆 Best Overall</h4>
            <p><strong>LightGBM</strong></p>
            <ul>
                <li>High accuracy (84.6%)</li>
                <li>Fast inference (0.05s)</li>
                <li>Small model size (5MB)</li>
                <li>Production ready</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea22 0%, #667eea44 100%); padding: 1rem; border-radius: 10px;">
            <h4>⚡ Fastest</h4>
            <p><strong>Baseline (LogReg)</strong></p>
            <ul>
                <li>Ultra-fast (0.01s)</li>
                <li>Tiny size (0.1MB)</li>
                <li>Good accuracy (87.8%)</li>
                <li>Resource efficient</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #764ba222 0%, #764ba244 100%); padding: 1rem; border-radius: 10px;">
            <h4>🎯 Most Accurate</h4>
            <p><strong>LSTM</strong></p>
            <ul>
                <li>Highest accuracy (92.3%)</li>
                <li>Good F1 score</li>
                <li>Attention mechanism</li>
                <li>Deep learning power</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)