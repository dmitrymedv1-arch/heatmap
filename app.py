import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io
import re
from typing import List, Optional, Tuple
import kaleido  # Для сохранения изображений

# Page configuration
st.set_page_config(
    page_title="Heatmap Generator",
    page_icon="🔥",
    layout="wide"
)

# Data processing functions
def preprocess_uploaded_content(content: str) -> str:
    """
    Preprocess uploaded content for handling incomplete cases
    """
    lines = content.strip().split('\n')
    processed_lines = []
    last_x_value = None
    
    for line in lines:
        # Remove extra spaces at the beginning and end of the line
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Split the line into parts (tab, comma, space)
        if '\t' in line:
            parts = line.split('\t')
        elif ',' in line:
            parts = line.split(',')
        else:
            # Split by spaces (consider multiple spaces)
            parts = re.split(r'\s+', line)
        
        # Remove empty elements
        parts = [p.strip() for p in parts if p.strip()]
        
        # Handle cases with missing X values
        if len(parts) == 1:
            # If only one value, it could be a new X
            last_x_value = parts[0]
            continue
        elif len(parts) == 2:
            # If two values, it could be Y and Value without X
            if last_x_value is not None:
                processed_lines.append(f"{last_x_value},{parts[0]},{parts[1]}")
            else:
                # If X is not previously defined, skip or use empty value
                continue
        elif len(parts) >= 3:
            # Complete string with X, Y and Value
            processed_lines.append(f"{parts[0]},{parts[1]},{parts[2]}")
            last_x_value = parts[0]
    
    return '\n'.join(processed_lines)

def parse_data(content: str) -> pd.DataFrame:
    """
    Parse data from string to DataFrame
    """
    # Preprocess data
    processed_content = preprocess_uploaded_content(content)
    
    # Read data
    try:
        # Try different delimiters
        for delimiter in [',', '\t', ' ']:
            try:
                # Try to read as CSV
                df = pd.read_csv(io.StringIO(processed_content), sep=delimiter, header=None, engine='python')
                if df.shape[1] >= 3:
                    df = df.iloc[:, :3]  # Take only first 3 columns
                    df.columns = ['X', 'Y', 'Value']
                    break
            except:
                continue
    except Exception as e:
        st.error(f"Error reading data: {e}")
        return None
    
    # Convert to string for categorical data
    df['X'] = df['X'].astype(str)
    df['Y'] = df['Y'].astype(str)
    
    # Try to convert Value to numeric format
    try:
        df['Value'] = pd.to_numeric(df['Value'])
    except:
        st.warning("Could not convert values to numeric format. Using strings.")
    
    return df

def create_pivot_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create pivot table for heatmap
    """
    if df is None or df.empty:
        return None
    
    # Create pivot table
    pivot_df = df.pivot(index='Y', columns='X', values='Value')
    
    # Sort indices for better display
    pivot_df = pivot_df.sort_index()
    pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)
    
    return pivot_df

def normalize_data(pivot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize data (0-1)
    """
    if pivot_df is None or pivot_df.empty:
        return None
    
    min_val = pivot_df.min().min()
    max_val = pivot_df.max().max()
    
    if max_val == min_val:
        return pivot_df
    
    # Normalization
    normalized_df = (pivot_df - min_val) / (max_val - min_val)
    return normalized_df

def create_smooth_contour(pivot_df: pd.DataFrame, colorscale: str = 'Viridis') -> go.Figure:
    """
    Create smooth contour plot (height map)
    """
    if pivot_df is None or pivot_df.empty:
        return None
    
    # Prepare data for contour plot
    x = list(range(len(pivot_df.columns)))
    y = list(range(len(pivot_df.index)))
    z = pivot_df.values
    
    fig = go.Figure(data=go.Contour(
        z=z,
        x=x,
        y=y,
        colorscale=colorscale,
        contours=dict(
            showlabels=True,
            labelfont=dict(size=12, color='black'),
        ),
        line=dict(width=0),  # Remove contour lines for smooth transition
        hoverongaps=False,
        colorbar=dict(
            title='Value',
            tickfont=dict(color='black')
        )
    ))
    
    # Configure axes
    fig.update_xaxes(
        ticktext=pivot_df.columns.tolist(),
        tickvals=x,
        title='X',
        tickfont=dict(color='black'),
        gridcolor='lightgray'
    )
    
    fig.update_yaxes(
        ticktext=pivot_df.index.tolist(),
        tickvals=y,
        title='Y',
        tickfont=dict(color='black'),
        gridcolor='lightgray'
    )
    
    fig.update_layout(
        title='Contour Map (smooth transition)',
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=600,
        height=500
    )
    
    return fig

# Main interface
st.title("🔥 Heatmap Generator for Scientific Publications")
st.markdown("""
Upload data in X,Y,Value format (comma, tab or space separated) or use example data.
""")

# Sidebar for settings
with st.sidebar:
    st.header("Plot Settings")
    
    # Axis settings
    st.subheader("Axis Settings")
    x_label = st.text_input("X-axis label", value="X")
    y_label = st.text_input("Y-axis label", value="Y")
    colorbar_title = st.text_input("Colorbar title", value="Value")
    
    # Font settings
    st.subheader("Font Settings")
    axis_font_size = st.slider("Axis font size", 8, 24, 14)
    tick_font_size = st.slider("Tick font size", 8, 20, 12)
    colorbar_font_size = st.slider("Colorbar font size", 8, 20, 12)
    
    # Display settings
    st.subheader("Data Display")
    show_values = st.checkbox("Show values in cells", value=True)
    value_format = st.selectbox("Value format", 
                               ["Auto", "Integer", "Two decimals", "Three decimals", "Scientific"])
    
    # Color palette selection
    st.subheader("Color Palette")
    
    # Built-in Plotly palettes
    builtin_palettes = [
        "Viridis", "Plasma", "Inferno", "Magma", "Cividis",
        "Greys", "RdBu", "RdYlBu", "Picnic", "Rainbow",
        "Portland", "Jet", "Hot", "Blackbody", "Electric"
    ]
    
    selected_palette = st.selectbox("Select palette", builtin_palettes, index=0)
    
    # Custom palette
    st.markdown("---")
    st.subheader("Custom Palette")
    use_custom_palette = st.checkbox("Use custom palette")
    
    custom_colors = []
    if use_custom_palette:
        color_count = st.slider("Number of colors in palette", 2, 10, 3)
        for i in range(color_count):
            color = st.color_picker(f"Color {i+1}", value="#%06x" % (i * 255 // color_count))
            custom_colors.append(color)
    
    # Additional plots settings
    st.markdown("---")
    st.subheader("Additional Plots")
    show_normalized = st.checkbox("Show normalized plot", value=True)
    show_contour = st.checkbox("Show contour map", value=True)

# Main area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Data Upload")
    
    # Example data
    example_choice = st.selectbox(
        "Select example data",
        ["Upload your own data", "Example 1: Simple", "Example 2: With gaps", 
         "Example 3: Numeric axes", "Example 4: Negative values"]
    )
    
    if example_choice == "Example 1: Simple":
        example_data = """X,Y,Value
A,Jan,10
A,Feb,20
B,Jan,15
B,Feb,25"""
    elif example_choice == "Example 2: With gaps":
        example_data = """A\t1\t0.2
\t2\t0.3
\t3\t0.4
B\t1\t0.25
\t2\t0.35
\t3\t0.45"""
    elif example_choice == "Example 3: Numeric axes":
        example_data = """X Y Value
1 1 0.5
1 2 0.7
2 1 0.3
2 2 0.9
3 1 0.6
3 2 0.4"""
    elif example_choice == "Example 4: Negative values":
        example_data = """X,Y,Value
A,Jan,-10
A,Feb,20
B,Jan,15
B,Feb,-5
C,Jan,30
C,Feb,-15"""
    else:
        example_data = ""
    
    # Data input field
    data_input = st.text_area(
        "Enter data (X, Y, Value separated by comma, tab or space):",
        value=example_data,
        height=200
    )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Or upload a file",
        type=['txt', 'csv', 'tsv', 'dat']
    )
    
    if uploaded_file is not None:
        content = uploaded_file.read().decode('utf-8')
        data_input = content
    
    # Process button
    if st.button("Generate Heatmaps", type="primary"):
        if data_input.strip():
            with st.spinner("Processing data..."):
                df = parse_data(data_input)
                
                if df is not None and not df.empty:
                    st.session_state.df = df
                    st.session_state.data_ready = True
                else:
                    st.error("Failed to process data. Please check the format.")
        else:
            st.warning("Please enter data or upload a file.")

with col2:
    st.header("Data Preview")
    
    if 'df' in st.session_state and st.session_state.get('data_ready', False):
        df = st.session_state.df
        
        st.subheader("Processed Data")
        st.dataframe(df, use_container_width=True)
        
        st.subheader("Data Statistics")
        col_stats1, col_stats2 = st.columns(2)
        with col_stats1:
            st.metric("Number of rows", len(df))
            st.metric("Unique X values", df['X'].nunique())
        with col_stats2:
            st.metric("Unique Y values", df['Y'].nunique())
            st.metric("Value range", 
                     f"{df['Value'].min():.2f} - {df['Value'].max():.2f}")
        
        st.subheader("Pivot Table")
        pivot_df = create_pivot_table(df)
        if pivot_df is not None:
            st.dataframe(pivot_df, use_container_width=True)

# Plots area
if 'df' in st.session_state and st.session_state.get('data_ready', False):
    st.markdown("---")
    st.header("Heatmaps")
    
    df = st.session_state.df
    pivot_df = create_pivot_table(df)
    
    if pivot_df is not None:
        # Value format configuration
        if value_format == "Integer":
            text_format = ".0f"
        elif value_format == "Two decimals":
            text_format = ".2f"
        elif value_format == "Three decimals":
            text_format = ".3f"
        elif value_format == "Scientific":
            text_format = ".2e"
        else:
            # Automatic format selection
            if df['Value'].dtype == np.int64:
                text_format = ".0f"
            else:
                text_format = ".2f"
        
        # Create color scale
        if use_custom_palette and custom_colors:
            # Custom color scale
            colorscale = [[i/(len(custom_colors)-1), color] for i, color in enumerate(custom_colors)]
        else:
            # Use built-in palette
            colorscale = selected_palette
        
        # 1. MAIN HEATMAP
        st.subheader("1. Main Heatmap")
        
        # Create text for cells
        if show_values:
            text_matrix = np.round(pivot_df.values, 
                                  0 if text_format == ".0f" else 
                                  2 if text_format == ".2f" else
                                  3 if text_format == ".3f" else 2)
            text_matrix = text_matrix.astype(str)
        else:
            text_matrix = None
        
        fig1 = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns.tolist(),
            y=pivot_df.index.tolist(),
            colorscale=colorscale,
            text=text_matrix,
            texttemplate='%{text}',
            hoverongaps=False,
            hoverinfo='x+y+z',
            colorbar=dict(
                title=colorbar_title,
                tickfont=dict(size=colorbar_font_size-2, color='black')
            ),
            xgap=1,
            ygap=1
        ))
        
        # Layout configuration for main plot
        fig1.update_layout(
            title=dict(
                text="Heatmap (with borders)",
                font=dict(size=16, color='black'),
                x=0.5
            ),
            xaxis=dict(
                title=dict(
                    text=x_label,
                    font=dict(size=axis_font_size, color='black')
                ),
                tickfont=dict(size=tick_font_size, color='black'),
                gridcolor='black',
                linecolor='black',
                mirror=True,
                showline=True,
                zeroline=False
            ),
            yaxis=dict(
                title=dict(
                    text=y_label,
                    font=dict(size=axis_font_size, color='black')
                ),
                tickfont=dict(size=tick_font_size, color='black'),
                gridcolor='black',
                linecolor='black',
                mirror=True,
                showline=True,
                zeroline=False
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=800,
            height=500,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # 2. NORMALIZED HEATMAP (only if all values are non-negative)
        if show_normalized and (pivot_df.values.min() >= 0):
            st.subheader("2. Normalized Heatmap (0-1)")
            
            normalized_df = normalize_data(pivot_df)
            
            if normalized_df is not None:
                # Create text for cells
                if show_values:
                    norm_text_matrix = np.round(normalized_df.values, 3).astype(str)
                else:
                    norm_text_matrix = None
                
                fig2 = go.Figure(data=go.Heatmap(
                    z=normalized_df.values,
                    x=normalized_df.columns.tolist(),
                    y=normalized_df.index.tolist(),
                    colorscale=colorscale,
                    text=norm_text_matrix,
                    texttemplate='%{text}',
                    hoverongaps=False,
                    hoverinfo='x+y+z',
                    colorbar=dict(
                        title="Normalized Value (0-1)",
                        tickfont=dict(size=colorbar_font_size-2, color='black')
                    ),
                    xgap=1,
                    ygap=1
                ))
                
                fig2.update_layout(
                    title=dict(
                        text="Normalized Heatmap",
                        font=dict(size=16, color='black'),
                        x=0.5
                    ),
                    xaxis=dict(
                        title=dict(
                            text=x_label,
                            font=dict(size=axis_font_size, color='black')
                        ),
                        tickfont=dict(size=tick_font_size, color='black')
                    ),
                    yaxis=dict(
                        title=dict(
                            text=y_label,
                            font=dict(size=axis_font_size, color='black')
                        ),
                        tickfont=dict(size=tick_font_size, color='black')
                    ),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    width=800,
                    height=500
                )
                
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Normalization not required or impossible (all values are equal)")
        elif show_normalized:
            st.info("Normalized plot not shown because there are negative values")
        
        # 3. CONTOUR MAP (smooth transition)
        if show_contour:
            st.subheader("3. Contour Map (smooth transition)")
            
            fig3 = create_smooth_contour(pivot_df, selected_palette)
            if fig3:
                # Update axis labels
                fig3.update_xaxes(title_text=x_label)
                fig3.update_yaxes(title_text=y_label)
                
                st.plotly_chart(fig3, use_container_width=True)
                
                # Additional contour map variants
                st.markdown("**Contour Map Variants:**")
                
                col_cont1, col_cont2 = st.columns(2)
                
                with col_cont1:
                    # Contour map with lines
                    fig3_lines = go.Figure(data=go.Contour(
                        z=pivot_df.values,
                        x=list(range(len(pivot_df.columns))),
                        y=list(range(len(pivot_df.index))),
                        colorscale=selected_palette,
                        contours=dict(
                            coloring='lines',
                            showlabels=True,
                            labelfont=dict(size=10, color='black')
                        ),
                        line=dict(width=2),
                        colorbar=dict(title='Value')
                    ))
                    
                    fig3_lines.update_xaxes(
                        ticktext=pivot_df.columns.tolist(),
                        tickvals=list(range(len(pivot_df.columns))),
                        title=x_label
                    )
                    fig3_lines.update_yaxes(
                        ticktext=pivot_df.index.tolist(),
                        tickvals=list(range(len(pivot_df.index))),
                        title=y_label
                    )
                    
                    fig3_lines.update_layout(
                        title='Contour Map (with lines)',
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        height=400
                    )
                    
                    st.plotly_chart(fig3_lines, use_container_width=True)
                
                with col_cont2:
                    # 3D Surface plot
                    if len(pivot_df.columns) > 1 and len(pivot_df.index) > 1:
                        fig3_surface = go.Figure(data=go.Surface(
                            z=pivot_df.values,
                            colorscale=selected_palette,
                            contours=dict(
                                z=dict(
                                    show=True,
                                    usecolormap=True,
                                    highlightcolor="limegreen",
                                    project=dict(z=True)
                                )
                            ),
                            colorbar=dict(title='Value')
                        ))
                        
                        fig3_surface.update_layout(
                            title='3D Surface Plot',
                            scene=dict(
                                xaxis=dict(title=x_label, ticktext=pivot_df.columns.tolist()),
                                yaxis=dict(title=y_label, ticktext=pivot_df.index.tolist()),
                                zaxis=dict(title=colorbar_title),
                                aspectmode='manual',
                                aspectratio=dict(x=1, y=1, z=0.7)
                            ),
                            width=600,
                            height=500,
                            margin=dict(l=0, r=0, b=0, t=30)
                        )
                        
                        st.plotly_chart(fig3_surface, use_container_width=True)
        
        # Export options
        st.markdown("---")
        st.subheader("Export Options")
        
        col_export1, col_export2, col_export3 = st.columns(3)
        
        with col_export1:
            if st.button("Save All Plots"):
                try:
                    # Save plots using Streamlit's download functionality
                    import base64
                    
                    # Convert figures to bytes
                    img1_bytes = fig1.to_image(format="png")
                    img2_bytes = fig2.to_image(format="png") if 'fig2' in locals() and show_normalized and pivot_df.values.min() >= 0 else None
                    img3_bytes = fig3.to_image(format="png") if 'fig3' in locals() and show_contour else None
                    
                    # Create download buttons
                    st.download_button(
                        label="Download Main Heatmap",
                        data=img1_bytes,
                        file_name="heatmap_main.png",
                        mime="image/png"
                    )
                    
                    if img2_bytes:
                        st.download_button(
                            label="Download Normalized Heatmap",
                            data=img2_bytes,
                            file_name="heatmap_normalized.png",
                            mime="image/png"
                        )
                    
                    if img3_bytes:
                        st.download_button(
                            label="Download Contour Map",
                            data=img3_bytes,
                            file_name="contour_map.png",
                            mime="image/png"
                        )
                    
                    st.success("Plots are ready for download")
                    
                except Exception as e:
                    st.error(f"Error saving plots: {e}")
                    st.info("Note: Plot saving requires the kaleido package. Install it with: pip install kaleido")
                    
        with col_export2:
            # Export data
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Data (CSV)",
                data=csv,
                file_name="heatmap_data.csv",
                mime="text/csv"
            )
            
        with col_export3:
            # Export pivot table
            pivot_csv = pivot_df.to_csv()
            st.download_button(
                label="Download Pivot Table",
                data=pivot_csv,
                file_name="pivot_table.csv",
                mime="text/csv"
            )

# Data format information
with st.expander("📋 Data Format Information"):
    st.markdown("""
    ### Supported Data Formats:
    
    1. **CSV format**: X,Y,Value separated by comma
    ```
    A,Jan,10
    A,Feb,20
    B,Jan,15
    B,Feb,25
    ```
    
    2. **TSV format**: X,Y,Value separated by tab
    ```
    A	Jan	10
    A	Feb	20
    B	Jan	15
    B	Feb	25
    ```
    
    3. **Space separated**: X Y Value separated by space
    ```
    A Jan 10
    A Feb 20
    B Jan 15
    B Feb 25
    ```
    
    ### Handling Incomplete Data:
    
    The app automatically handles data with missing X values:
    
    **Input data:**
    ```
    A	
    1	0.2
    2	0.3
    3	0.4
    B	
    1	0.25
    2	0.35
    3	0.45
    ```
    
    **Will be converted to:**
    ```
    A,1,0.2
    A,2,0.3
    A,3,0.4
    B,1,0.25
    B,2,0.35
    B,3,0.45
    ```
    
    ### Plot Types:
    
    1. **Main Heatmap** - Classic heatmap with clear borders
    2. **Normalized Heatmap** - Values normalized to 0-1 range
    3. **Contour Map** - Smooth transition between values (height map)
    """)

# Footer
st.markdown("---")
st.markdown("""
**Heatmap Generator for Scientific Publications** | Optimized for research papers
""")

# Add installation instructions in an expander
with st.expander("🛠️ Installation & Troubleshooting"):
    st.markdown("""
    ### To enable plot saving functionality:
    
    1. **Install kaleido package:**
    ```bash
    pip install kaleido
    ```
    
    2. **If you encounter errors when saving plots:**
    - Make sure kaleido is installed
    - Try restarting the app
    - On Streamlit Cloud, kaleido should be included in requirements.txt
    
    ### Requirements:
    ```txt
    streamlit>=1.28.0
    pandas>=2.0.0
    numpy>=1.24.0
    plotly>=5.17.0
    kaleido>=0.2.0
    ```
    """)
