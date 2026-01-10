import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io
import re
from typing import List, Optional, Tuple

# Настройка страницы
st.set_page_config(
    page_title="Heatmap Generator",
    page_icon="🔥",
    layout="wide"
)

# Функции для обработки данных
def preprocess_uploaded_content(content: str) -> str:
    """
    Предобработка загруженного контента для обработки неполных случаев
    """
    lines = content.strip().split('\n')
    processed_lines = []
    last_x_value = None
    
    for line in lines:
        # Удаляем лишние пробелы в начале и конце строки
        line = line.strip()
        
        # Пропускаем пустые строки
        if not line:
            continue
            
        # Разделяем строку на части (табуляция, запятая, пробел)
        if '\t' in line:
            parts = line.split('\t')
        elif ',' in line:
            parts = line.split(',')
        else:
            # Разделение по пробелам (учитываем множественные пробелы)
            parts = re.split(r'\s+', line)
        
        # Удаляем пустые элементы
        parts = [p.strip() for p in parts if p.strip()]
        
        # Обработка случаев с недостающими значениями X
        if len(parts) == 1:
            # Если только одно значение, это может быть новый X
            last_x_value = parts[0]
            continue
        elif len(parts) == 2:
            # Если два значения, это может быть Y и Value без X
            if last_x_value is not None:
                processed_lines.append(f"{last_x_value},{parts[0]},{parts[1]}")
            else:
                # Если X не определен ранее, пропускаем или используем пустое значение
                continue
        elif len(parts) >= 3:
            # Полная строка с X, Y и Value
            processed_lines.append(f"{parts[0]},{parts[1]},{parts[2]}")
            last_x_value = parts[0]
    
    return '\n'.join(processed_lines)

def parse_data(content: str) -> pd.DataFrame:
    """
    Парсинг данных из строки в DataFrame
    """
    # Предобработка данных
    processed_content = preprocess_uploaded_content(content)
    
    # Чтение данных
    try:
        # Пробуем разные разделители
        for delimiter in [',', '\t', ' ']:
            try:
                # Пробуем прочитать как CSV
                df = pd.read_csv(io.StringIO(processed_content), sep=delimiter, header=None, engine='python')
                if df.shape[1] >= 3:
                    df = df.iloc[:, :3]  # Берем только первые 3 столбца
                    df.columns = ['X', 'Y', 'Value']
                    break
            except:
                continue
    except Exception as e:
        st.error(f"Ошибка при чтении данных: {e}")
        return None
    
    # Преобразуем числовые значения
    df['X'] = df['X'].astype(str)
    df['Y'] = df['Y'].astype(str)
    
    # Пробуем преобразовать Value в числовой формат
    try:
        df['Value'] = pd.to_numeric(df['Value'])
    except:
        st.warning("Не удалось преобразовать значения в числовой формат. Используются строки.")
    
    return df

def create_pivot_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создание сводной таблицы для тепловой карты
    """
    if df is None or df.empty:
        return None
    
    # Создаем сводную таблицу
    pivot_df = df.pivot(index='Y', columns='X', values='Value')
    
    # Сортируем индексы для лучшего отображения
    pivot_df = pivot_df.sort_index()
    pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)
    
    return pivot_df

def normalize_data(pivot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Нормировка данных (0-1)
    """
    if pivot_df is None or pivot_df.empty:
        return None
    
    min_val = pivot_df.min().min()
    max_val = pivot_df.max().max()
    
    if max_val == min_val:
        return pivot_df
    
    # Нормировка
    normalized_df = (pivot_df - min_val) / (max_val - min_val)
    return normalized_df

def create_smooth_contour(pivot_df: pd.DataFrame) -> go.Figure:
    """
    Создание плавного контурного графика (карта высот)
    """
    if pivot_df is None or pivot_df.empty:
        return None
    
    # Преобразуем данные для контурного графика
    x = list(range(len(pivot_df.columns)))
    y = list(range(len(pivot_df.index)))
    z = pivot_df.values
    
    fig = go.Figure(data=go.Contour(
        z=z,
        x=x,
        y=y,
        colorscale='Viridis',
        contours=dict(
            showlabels=True,
            labelfont=dict(size=12, color='black'),
        ),
        line=dict(width=0),  # Убираем линии контуров для плавного перехода
        hoverongaps=False
    ))
    
    # Добавляем colorbar отдельно через update_layout
    fig.update_layout(
        coloraxis_colorbar=dict(
            title='Значение',
            titleside='right',
            tickfont=dict(color='black')
        )
    )
    
    # Настройка осей
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
        title='Контурная карта (плавный переход)',
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=600,
        height=500
    )
    
    return fig

# Основной интерфейс
st.title("🔥 Генератор тепловых карт для научных публикаций")
st.markdown("""
Загрузите данные в формате X,Y,Value (через запятую, табуляцию или пробел) или используйте примеры данных.
""")

# Боковая панель для настроек
with st.sidebar:
    st.header("Настройки графиков")
    
    # Настройки осей
    st.subheader("Настройки осей")
    x_label = st.text_input("Название оси X", value="X")
    y_label = st.text_input("Название оси Y", value="Y")
    colorbar_title = st.text_input("Название шкалы", value="Значение")
    
    # Настройки шрифтов
    st.subheader("Настройки шрифтов")
    axis_font_size = st.slider("Размер шрифта осей", 8, 24, 14)
    tick_font_size = st.slider("Размер шрифта значений", 8, 20, 12)
    colorbar_font_size = st.slider("Размер шрифта шкалы", 8, 20, 12)
    
    # Настройки отображения
    st.subheader("Отображение данных")
    show_values = st.checkbox("Показывать значения в ячейках", value=True)
    value_format = st.selectbox("Формат значений", 
                               ["Авто", "Целые числа", "Два знака", "Три знака", "Научный"])
    
    # Выбор цветовой палитры
    st.subheader("Цветовая палитра")
    
    # Встроенные палитры Plotly
    builtin_palettes = [
        "Viridis", "Plasma", "Inferno", "Magma", "Cividis",
        "Greys", "RdBu", "RdYlBu", "Picnic", "Rainbow",
        "Portland", "Jet", "Hot", "Blackbody", "Electric"
    ]
    
    selected_palette = st.selectbox("Выберите палитру", builtin_palettes, index=0)
    
    # Пользовательская палитра
    st.markdown("---")
    st.subheader("Пользовательская палитра")
    use_custom_palette = st.checkbox("Использовать собственную палитру")
    
    custom_colors = []
    if use_custom_palette:
        color_count = st.slider("Количество цветов в палитре", 2, 10, 3)
        for i in range(color_count):
            color = st.color_picker(f"Цвет {i+1}", value="#%06x" % (i * 255 // color_count))
            custom_colors.append(color)
    
    # Настройки дополнительных графиков
    st.markdown("---")
    st.subheader("Дополнительные графики")
    show_normalized = st.checkbox("Показать нормированный график", value=True)
    show_contour = st.checkbox("Показать контурную карту", value=True)

# Основная область
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Загрузка данных")
    
    # Примеры данных
    example_choice = st.selectbox(
        "Выберите пример данных",
        ["Загрузите свои данные", "Пример 1: Простой", "Пример 2: С пропусками", 
         "Пример 3: Числовые оси", "Пример 4: Отрицательные значения"]
    )
    
    if example_choice == "Пример 1: Простой":
        example_data = """X,Y,Value
A,Jan,10
A,Feb,20
B,Jan,15
B,Feb,25"""
    elif example_choice == "Пример 2: С пропусками":
        example_data = """A\t1\t0.2
\t2\t0.3
\t3\t0.4
B\t1\t0.25
\t2\t0.35
\t3\t0.45"""
    elif example_choice == "Пример 3: Числовые оси":
        example_data = """X Y Value
1 1 0.5
1 2 0.7
2 1 0.3
2 2 0.9
3 1 0.6
3 2 0.4"""
    elif example_choice == "Пример 4: Отрицательные значения":
        example_data = """X,Y,Value
A,Jan,-10
A,Feb,20
B,Jan,15
B,Feb,-5
C,Jan,30
C,Feb,-15"""
    else:
        example_data = ""
    
    # Поле для ввода данных
    data_input = st.text_area(
        "Введите данные (X, Y, Value через запятую, табуляцию или пробел):",
        value=example_data,
        height=200
    )
    
    # Загрузка файла
    uploaded_file = st.file_uploader(
        "Или загрузите файл",
        type=['txt', 'csv', 'tsv', 'dat']
    )
    
    if uploaded_file is not None:
        content = uploaded_file.read().decode('utf-8')
        data_input = content
    
    # Кнопка обработки
    if st.button("Создать тепловые карты", type="primary"):
        if data_input.strip():
            with st.spinner("Обработка данных..."):
                df = parse_data(data_input)
                
                if df is not None and not df.empty:
                    st.session_state.df = df
                    st.session_state.data_ready = True
                else:
                    st.error("Не удалось обработать данные. Проверьте формат.")
        else:
            st.warning("Пожалуйста, введите данные или загрузите файл.")

with col2:
    st.header("Предварительный просмотр данных")
    
    if 'df' in st.session_state and st.session_state.get('data_ready', False):
        df = st.session_state.df
        
        st.subheader("Обработанные данные")
        st.dataframe(df, use_container_width=True)
        
        st.subheader("Статистика данных")
        col_stats1, col_stats2 = st.columns(2)
        with col_stats1:
            st.metric("Количество строк", len(df))
            st.metric("Уникальных X", df['X'].nunique())
        with col_stats2:
            st.metric("Уникальных Y", df['Y'].nunique())
            st.metric("Диапазон значений", 
                     f"{df['Value'].min():.2f} - {df['Value'].max():.2f}")
        
        st.subheader("Сводная таблица")
        pivot_df = create_pivot_table(df)
        if pivot_df is not None:
            st.dataframe(pivot_df, use_container_width=True)

# Область графиков
if 'df' in st.session_state and st.session_state.get('data_ready', False):
    st.markdown("---")
    st.header("Тепловые карты")
    
    df = st.session_state.df
    pivot_df = create_pivot_table(df)
    
    if pivot_df is not None:
        # Настройка формата значений
        if value_format == "Целые числа":
            text_format = ".0f"
        elif value_format == "Два знака":
            text_format = ".2f"
        elif value_format == "Три знака":
            text_format = ".3f"
        elif value_format == "Научный":
            text_format = ".2e"
        else:
            # Автоматический выбор формата
            if df['Value'].dtype == np.int64:
                text_format = ".0f"
            else:
                text_format = ".2f"
        
        # Создание цветовой шкалы
        if use_custom_palette and custom_colors:
            # Пользовательская цветовая шкала
            colorscale = [[i/(len(custom_colors)-1), color] for i, color in enumerate(custom_colors)]
        else:
            # Использование встроенной палитры
            colorscale = selected_palette
        
        # 1. ОСНОВНАЯ ТЕПЛОВАЯ КАРТА
        st.subheader("1. Основная тепловая карта")
        
        # Создание текста для ячеек
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
                title=dict(
                    text=colorbar_title,
                    font=dict(size=colorbar_font_size, color='black')
                ),
                tickfont=dict(size=colorbar_font_size-2, color='black')
            ),
            xgap=1,
            ygap=1
        ))
        
        # Настройка макета для основного графика
        fig1.update_layout(
            title=dict(
                text="Тепловая карта (с границами)",
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
        
        # 2. НОРМИРОВАННАЯ ТЕПЛОВАЯ КАРТА (только если все значения неотрицательны)
        if show_normalized and (pivot_df.values.min() >= 0):
            st.subheader("2. Нормированная тепловая карта (0-1)")
            
            normalized_df = normalize_data(pivot_df)
            
            if normalized_df is not None:
                # Создание текста для ячеек
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
                        title=dict(
                            text="Нормированное значение (0-1)",
                            font=dict(size=colorbar_font_size, color='black')
                        ),
                        tickfont=dict(size=colorbar_font_size-2, color='black')
                    ),
                    xgap=1,
                    ygap=1
                ))
                
                fig2.update_layout(
                    title=dict(
                        text="Нормированная тепловая карта",
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
                st.info("Нормировка не требуется или невозможна (все значения одинаковы)")
        elif show_normalized:
            st.info("Нормированный график не показан, так как есть отрицательные значения")
        
        # 3. КОНТУРНАЯ КАРТА (плавный переход)
        if show_contour:
            st.subheader("3. Контурная карта (плавный переход)")
            
            fig3 = create_smooth_contour(pivot_df)
            if fig3:
                # Обновляем названия осей
                fig3.update_xaxes(title_text=x_label)
                fig3.update_yaxes(title_text=y_label)
                
                st.plotly_chart(fig3, use_container_width=True)
                
                # Дополнительные варианты контурной карты
                st.markdown("**Варианты контурной карты:**")
                
                col_cont1, col_cont2 = st.columns(2)
                
                with col_cont1:
                    # Контурная карта с линиями
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
                        line=dict(width=2)
                    ))
                    
                    fig3_lines.update_layout(
                        coloraxis_colorbar=dict(
                            title='Значение'
                        )
                    )
                    
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
                        title='Контурная карта (с линиями)',
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        height=400
                    )
                    
                    st.plotly_chart(fig3_lines, use_container_width=True)
                
                with col_cont2:
                    # 3D поверхность
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
                            )
                        ))
                        
                        fig3_surface.update_layout(
                            title='3D поверхность',
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
        
        # Опции экспорта
        st.markdown("---")
        st.subheader("Экспорт графиков")
        
        col_export1, col_export2, col_export3 = st.columns(3)
        
        with col_export1:
            if st.button("Сохранить все графики"):
                # Сохранение графиков
                fig1.write_image("heatmap_main.png")
                if show_normalized and (pivot_df.values.min() >= 0) and normalized_df is not None:
                    fig2.write_image("heatmap_normalized.png")
                if show_contour and fig3:
                    fig3.write_image("contour_map.png")
                st.success("Графики сохранены в PNG формате")
                
        with col_export2:
            # Экспорт данных
            csv = df.to_csv(index=False)
            st.download_button(
                label="Скачать данные (CSV)",
                data=csv,
                file_name="heatmap_data.csv",
                mime="text/csv"
            )
            
        with col_export3:
            # Экспорт сводной таблицы
            pivot_csv = pivot_df.to_csv()
            st.download_button(
                label="Скачать сводную таблицу",
                data=pivot_csv,
                file_name="pivot_table.csv",
                mime="text/csv"
            )

# Информация о формате данных
with st.expander("📋 Информация о формате данных"):
    st.markdown("""
    ### Поддерживаемые форматы данных:
    
    1. **CSV формат**: X,Y,Value через запятую
    ```
    A,Jan,10
    A,Feb,20
    B,Jan,15
    B,Feb,25
    ```
    
    2. **TSV формат**: X,Y,Value через табуляцию
    ```
    A	Jan	10
    A	Feb	20
    B	Jan	15
    B	Feb	25
    ```
    
    3. **Пробелы**: X Y Value через пробел
    ```
    A Jan 10
    A Feb 20
    B Jan 15
    B Feb 25
    ```
    
    ### Обработка неполных данных:
    
    Приложение автоматически обрабатывает данные с пропущенными значениями X:
    
    **Входные данные:**
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
    
    **Будут преобразованы в:**
    ```
    A,1,0.2
    A,2,0.3
    A,3,0.4
    B,1,0.25
    B,2,0.35
    B,3,0.45
    ```
    
    ### Типы графиков:
    
    1. **Основная тепловая карта** - классическая heatmap с четкими границами
    2. **Нормированная тепловая карта** - значения преобразованы к диапазону 0-1
    3. **Контурная карта** - плавный переход между значениями (карта высот)
    """)

# Футер
st.markdown("---")
st.markdown("""
**Приложение для генерации тепловых карт** | Оптимизировано для научных публикаций
""")

