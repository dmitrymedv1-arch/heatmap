import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io
import re
import zipfile
import matplotlib.pyplot as plt
import matplotlib
from typing import List, Optional, Tuple

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
    
    # Keep original types but ensure they are strings for categorical representation
    df['X'] = df['X'].astype(str)
    df['Y'] = df['Y'].astype(str)
    
    # Store original values as separate columns for numeric sorting
    df['X_numeric'] = pd.to_numeric(df['X'], errors='coerce')
    df['Y_numeric'] = pd.to_numeric(df['Y'], errors='coerce')
    
    # Try to convert Value to numeric format
    try:
        df['Value'] = pd.to_numeric(df['Value'])
    except:
        st.warning("Could not convert values to numeric format. Using strings.")
    
    return df

def create_pivot_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create pivot table for heatmap - preserve original order
    """
    if df is None or df.empty:
        return None
    
    # Get unique X and Y values in the order they appear
    x_order = df['X'].unique().tolist()
    y_order = df['Y'].unique().tolist()
    
    # Try to sort numerically if possible
    if df['X_numeric'].notna().all():
        # Sort X values numerically
        x_sorted = df[['X', 'X_numeric']].drop_duplicates().sort_values('X_numeric')
        x_order = x_sorted['X'].tolist()
    
    if df['Y_numeric'].notna().all():
        # Sort Y values numerically
        y_sorted = df[['Y', 'Y_numeric']].drop_duplicates().sort_values('Y_numeric')
        y_order = y_sorted['Y'].tolist()
    
    # Create pivot table with specified order
    pivot_df = df.pivot(index='Y', columns='X', values='Value')
    
    # Reindex to preserve order
    pivot_df = pivot_df.reindex(index=y_order, columns=x_order)
    
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

def create_smooth_contour(pivot_df: pd.DataFrame, colorscale: str = 'viridis') -> go.Figure:
    """
    Create smooth contour plot (height map)
    """
    if pivot_df is None or pivot_df.empty:
        return None
    
    # Prepare data for contour plot
    x = list(range(len(pivot_df.columns)))
    y = list(range(len(pivot_df.index)))
    z = pivot_df.values
    
    # Create figure
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
        gridcolor='lightgray',
        type='category'  # Treat as categorical to preserve order
    )
    
    fig.update_yaxes(
        ticktext=pivot_df.index.tolist(),
        tickvals=y,
        title='Y',
        tickfont=dict(color='black'),
        gridcolor='lightgray',
        type='category'  # Treat as categorical to preserve order
    )
    
    fig.update_layout(
        title='Contour Map (smooth transition)',
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=600,
        height=500
    )
    
    return fig

def save_all_plots_matplotlib(pivot_df, normalized_df, x_label, y_label, dpi=300, show_values=True):
    """Save all plots using matplotlib"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Convert index and columns to numeric if possible for proper sorting
        try:
            # Try to convert to numeric for sorting
            x_vals = pd.to_numeric(pivot_df.columns, errors='ignore')
            y_vals = pd.to_numeric(pivot_df.index, errors='ignore')
            
            # If conversion successful, sort numerically
            if not isinstance(x_vals[0], str):
                col_order = np.argsort(x_vals)
                cols_sorted = [pivot_df.columns[i] for i in col_order]
            else:
                cols_sorted = pivot_df.columns.tolist()
                
            if not isinstance(y_vals[0], str):
                row_order = np.argsort(y_vals)
                rows_sorted = [pivot_df.index[i] for i in row_order]
            else:
                rows_sorted = pivot_df.index.tolist()
                
            pivot_df_sorted = pivot_df.loc[rows_sorted, cols_sorted]
        except:
            pivot_df_sorted = pivot_df
        
        # 1. Main Heatmap
        fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
        
        # Create heatmap
        im = ax.imshow(pivot_df_sorted.values, aspect='auto', cmap='viridis', 
                      extent=[0, len(pivot_df_sorted.columns), 0, len(pivot_df_sorted.index)])
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(pivot_df_sorted.columns)) + 0.5)
        ax.set_yticks(np.arange(len(pivot_df_sorted.index)) + 0.5)
        ax.set_xticklabels(pivot_df_sorted.columns.tolist())
        ax.set_yticklabels(pivot_df_sorted.index.tolist())
        
        # Rotate x labels for better visibility
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Value', rotation=270, labelpad=20)
        
        # Set labels
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title('Main Heatmap')
        
        # Add grid
        ax.grid(False)
        
        # Add values to cells if requested
        if show_values:
            for i in range(len(pivot_df_sorted.index)):
                for j in range(len(pivot_df_sorted.columns)):
                    text = ax.text(j + 0.5, i + 0.5, f'{pivot_df_sorted.values[i, j]:.2f}',
                                  ha="center", va="center", color="w", fontsize=8)
        
        plt.tight_layout()
        
        # Save to buffer
        heatmap_buffer = io.BytesIO()
        fig.savefig(heatmap_buffer, format='png', dpi=dpi, bbox_inches='tight')
        zip_file.writestr('heatmap_main.png', heatmap_buffer.getvalue())
        plt.close(fig)
        
        # 2. Normalized Heatmap (if available)
        if normalized_df is not None:
            try:
                # Sort normalized_df in same order
                normalized_df_sorted = normalized_df.loc[rows_sorted, cols_sorted]
            except:
                normalized_df_sorted = normalized_df
                
            fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
            im = ax.imshow(normalized_df_sorted.values, aspect='auto', cmap='viridis', 
                          vmin=0, vmax=1,
                          extent=[0, len(normalized_df_sorted.columns), 0, len(normalized_df_sorted.index)])
            
            # Set ticks and labels
            ax.set_xticks(np.arange(len(normalized_df_sorted.columns)) + 0.5)
            ax.set_yticks(np.arange(len(normalized_df_sorted.index)) + 0.5)
            ax.set_xticklabels(normalized_df_sorted.columns.tolist())
            ax.set_yticklabels(normalized_df_sorted.index.tolist())
            
            # Rotate x labels for better visibility
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Normalized Value (0-1)', rotation=270, labelpad=20)
            
            # Set labels
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title('Normalized Heatmap (0-1)')
            
            # Add grid
            ax.grid(False)
            
            # Add values to cells if requested
            if show_values:
                for i in range(len(normalized_df_sorted.index)):
                    for j in range(len(normalized_df_sorted.columns)):
                        text = ax.text(j + 0.5, i + 0.5, f'{normalized_df_sorted.values[i, j]:.3f}',
                                      ha="center", va="center", color="w", fontsize=8)
            
            plt.tight_layout()
            
            # Save to buffer
            norm_buffer = io.BytesIO()
            fig.savefig(norm_buffer, format='png', dpi=dpi, bbox_inches='tight')
            zip_file.writestr('heatmap_normalized.png', norm_buffer.getvalue())
            plt.close(fig)
        
        # 3. Contour Plot
        fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
        
        # Create meshgrid for contour plot
        X, Y = np.meshgrid(np.arange(len(pivot_df_sorted.columns)) + 0.5, 
                          np.arange(len(pivot_df_sorted.index)) + 0.5)
        
        # Create contour plot
        contour = ax.contourf(X, Y, pivot_df_sorted.values, cmap='viridis', levels=20)
        
        # Add contour lines
        ax.contour(X, Y, pivot_df_sorted.values, colors='black', linewidths=0.5, levels=10)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(pivot_df_sorted.columns)) + 0.5)
        ax.set_yticks(np.arange(len(pivot_df_sorted.index)) + 0.5)
        ax.set_xticklabels(pivot_df_sorted.columns.tolist())
        ax.set_yticklabels(pivot_df_sorted.index.tolist())
        
        # Rotate x labels for better visibility
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Value', rotation=270, labelpad=20)
        
        # Set labels
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title('Contour Plot')
        
        plt.tight_layout()
        
        # Save to buffer
        contour_buffer = io.BytesIO()
        fig.savefig(contour_buffer, format='png', dpi=dpi, bbox_inches='tight')
        zip_file.writestr('contour_plot.png', contour_buffer.getvalue())
        plt.close(fig)
        
        # 4. Data table as image
        fig, ax = plt.subplots(figsize=(12, 8), dpi=dpi)
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table_data = pivot_df_sorted.values.round(3)
        row_labels = pivot_df_sorted.index.tolist()
        col_labels = pivot_df_sorted.columns.tolist()
        
        table = ax.table(cellText=table_data,
                        rowLabels=row_labels,
                        colLabels=col_labels,
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        
        ax.set_title('Data Table', fontsize=16, pad=20)
        
        plt.tight_layout()
        
        # Save to buffer
        table_buffer = io.BytesIO()
        fig.savefig(table_buffer, format='png', dpi=dpi, bbox_inches='tight')
        zip_file.writestr('data_table.png', table_buffer.getvalue())
        plt.close(fig)
    
    return zip_buffer

# Main interface
st.title("🔥 Heatmap Generator for Scientific Publications")
st.markdown("""
Upload data in X,Y,Value format (comma, tab or space separated) or use example data.
Values will be displayed in their original order.
""")

# Sidebar for settings
with st.sidebar:
    st.header("Plot Settings")
    
    # Axis settings
    st.subheader("Axis Settings")
    x_label = st.text_input("X-axis label", value="X")
    y_label = st.text_input("Y-axis label", value="Y")
    colorbar_title = st.text_input("Colorbar title", value="Value")
    
    # Display options
    st.subheader("Display Options")
    sort_numerically = st.checkbox("Sort axes numerically", value=True, 
                                   help="If unchecked, axes will be displayed in the order they appear in the data")
    
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
    
    # Save settings
    st.markdown("---")
    st.subheader("Save Settings")
    save_dpi = st.slider("DPI for saving", 100, 600, 300)

# Main area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Data Upload")
    
    # Example data
    example_choice = st.selectbox(
        "Select example data",
        ["Upload your own data", "Example 1: Simple", "Example 2: Temperature data", 
         "Example 3: With gaps", "Example 4: Negative values"]
    )
    
    if example_choice == "Example 1: Simple":
        example_data = """X,Y,Value
A,Jan,10
A,Feb,20
B,Jan,15
B,Feb,25"""
    elif example_choice == "Example 2: Temperature data":
        example_data = """X,Y,Value
Method1,25,0.1
Method1,50,0.2
Method1,100,0.3
Method1,150,0.4
Method1,200,0.5
Method1,250,0.6
Method1,300,0.7
Method1,350,0.8
Method1,400,0.9
Method1,450,1.0
Method1,500,1.1
Method1,550,1.2
Method1,600,1.3
Method1,650,1.4
Method1,700,1.5
Method1,750,1.6
Method1,800,1.7
Method1,850,1.8
Method1,900,1.9
Method1,950,2.0
Method1,1000,2.1
Method2,25,0.15
Method2,50,0.25
Method2,100,0.35
Method2,150,0.45
Method2,200,0.55
Method2,250,0.65
Method2,300,0.75
Method2,350,0.85
Method2,400,0.95
Method2,450,1.05
Method2,500,1.15
Method2,550,1.25
Method2,600,1.35
Method2,650,1.45
Method2,700,1.55
Method2,750,1.65
Method2,800,1.75
Method2,850,1.85
Method2,900,1.95
Method2,950,2.05
Method2,1000,2.15"""
    elif example_choice == "Example 3: With gaps":
        example_data = """A\t1\t0.2
\t2\t0.3
\t3\t0.4
B\t1\t0.25
\t2\t0.35
\t3\t0.45"""
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
                    st.session_state.show_values = show_values
                    st.session_state.sort_numerically = sort_numerically
                else:
                    st.error("Failed to process data. Please check the format.")
        else:
            st.warning("Please enter data or upload a file.")

with col2:
    st.header("Data Preview")
    
    if 'df' in st.session_state and st.session_state.get('data_ready', False):
        df = st.session_state.df
        
        st.subheader("Processed Data")
        st.dataframe(df[['X', 'Y', 'Value']], use_container_width=True)
        
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
            # Apply numeric sorting if requested
            if sort_numerically and 'X_numeric' in df.columns and 'Y_numeric' in df.columns:
                try:
                    # Sort columns numerically
                    x_sorted = df[['X', 'X_numeric']].drop_duplicates().sort_values('X_numeric')
                    y_sorted = df[['Y', 'Y_numeric']].drop_duplicates().sort_values('Y_numeric')
                    
                    # Reindex pivot table
                    pivot_df = pivot_df.reindex(index=y_sorted['Y'], columns=x_sorted['X'])
                except:
                    pass
            
            st.dataframe(pivot_df, use_container_width=True)
            
            # Show order information
            st.info(f"X-axis order: {', '.join(pivot_df.columns.tolist()[:5])}{'...' if len(pivot_df.columns) > 5 else ''}")
            st.info(f"Y-axis order: {', '.join(pivot_df.index.tolist()[:5])}{'...' if len(pivot_df.index) > 5 else ''}")

# Plots area
if 'df' in st.session_state and st.session_state.get('data_ready', False):
    st.markdown("---")
    st.header("Heatmaps")
    
    df = st.session_state.df
    pivot_df = create_pivot_table(df)
    
    if pivot_df is not None:
        # Apply numeric sorting if requested for plots
        if sort_numerically and 'X_numeric' in df.columns and 'Y_numeric' in df.columns:
            try:
                # Sort columns numerically
                x_sorted = df[['X', 'X_numeric']].drop_duplicates().sort_values('X_numeric')
                y_sorted = df[['Y', 'Y_numeric']].drop_duplicates().sort_values('Y_numeric')
                
                # Reindex pivot table
                pivot_df = pivot_df.reindex(index=y_sorted['Y'], columns=x_sorted['X'])
            except:
                pass
        
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
                type='category',  # Keep as categorical to preserve order
                tickmode='array',
                tickvals=list(range(len(pivot_df.columns))),
                ticktext=pivot_df.columns.tolist()
            ),
            yaxis=dict(
                title=dict(
                    text=y_label,
                    font=dict(size=axis_font_size, color='black')
                ),
                tickfont=dict(size=tick_font_size, color='black'),
                type='category',  # Keep as categorical to preserve order
                tickmode='array',
                tickvals=list(range(len(pivot_df.index))),
                ticktext=pivot_df.index.tolist()
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=800,
            height=500,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # 2. NORMALIZED HEATMAP (only if all values are non-negative)
        normalized_df = None
        if show_normalized:
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
                        tickfont=dict(size=tick_font_size, color='black'),
                        type='category',
                        tickmode='array',
                        tickvals=list(range(len(normalized_df.columns))),
                        ticktext=normalized_df.columns.tolist()
                    ),
                    yaxis=dict(
                        title=dict(
                            text=y_label,
                            font=dict(size=axis_font_size, color='black')
                        ),
                        tickfont=dict(size=tick_font_size, color='black'),
                        type='category',
                        tickmode='array',
                        tickvals=list(range(len(normalized_df.index))),
                        ticktext=normalized_df.index.tolist()
                    ),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    width=800,
                    height=500
                )
                
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Normalization not required or impossible (all values are equal)")
        
        # 3. CONTOUR MAP (smooth transition)
        if show_contour:
            st.subheader("3. Contour Map (smooth transition)")
            
            fig3 = create_smooth_contour(pivot_df, selected_palette)
            if fig3:
                # Update axis labels
                fig3.update_xaxes(title_text=x_label)
                fig3.update_yaxes(title_text=y_label)
                
                st.plotly_chart(fig3, use_container_width=True)
        
        # Export options
        st.markdown("---")
        st.subheader("Export Options")
        
        col_export1, col_export2, col_export3 = st.columns(3)
        
        with col_export1:
            if st.button("📦 Save All Plots (ZIP)"):
                try:
                    with st.spinner("Creating ZIP archive with high-resolution images..."):
                        # Create normalized_df if not already created
                        if normalized_df is None and show_normalized:
                            normalized_df = normalize_data(pivot_df)
                        
                        # Save all plots using matplotlib
                        zip_buffer = save_all_plots_matplotlib(
                            pivot_df, normalized_df, x_label, y_label, 
                            dpi=save_dpi, show_values=show_values
                        )
                        
                        # Create download button for ZIP
                        st.download_button(
                            label=f"Download ZIP ({save_dpi} DPI)",
                            data=zip_buffer.getvalue(),
                            file_name=f"heatmap_plots_{save_dpi}dpi.zip",
                            mime="application/zip"
                        )
                        
                        st.success(f"✅ All plots saved with {save_dpi} DPI resolution!")
                        
                except Exception as e:
                    st.error(f"Error saving plots: {str(e)}")
                    
        with col_export2:
            # Export data
            csv = df[['X', 'Y', 'Value']].to_csv(index=False)
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
    Temperature,Pressure,Value
    25,1,0.1
    50,1,0.2
    100,1,0.3
    150,1,0.4
    ```
    
    2. **TSV format**: X,Y,Value separated by tab
    ```
    Temperature	Pressure	Value
    25	1	0.1
    50	1	0.2
    100	1	0.3
    150	1	0.4
    ```
    
    3. **Space separated**: X Y Value separated by space
    ```
    Temperature Pressure Value
    25 1 0.1
    50 1 0.2
    100 1 0.3
    150 1 0.4
    ```
    
    ### Key Features:
    
    - **Preserves axis order**: Values are displayed in the order they appear in your data
    - **Numeric sorting option**: Optional automatic numeric sorting of axis values
    - **High-resolution export**: Save all plots as PNG images with configurable DPI (100-600)
    - **Multiple plot types**: Heatmaps, normalized plots, and contour maps
    """)

# Footer
st.markdown("---")
st.markdown("""
**Heatmap Generator for Scientific Publications** | Optimized for research papers
""")
