import tkinter as tk
from tkinter import ttk, messagebox
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from tkinter import filedialog


class LayerFrame(ttk.Frame):
    def __init__(self, parent, layer_num):
        super().__init__(parent)
        self.layer_num = layer_num

        # 레이어 프레임 설정
        ttk.Label(self, text=f"Layer {layer_num}:").grid(row=0, column=0, padx=5, pady=2)

        # 노드 수 설정
        ttk.Label(self, text="노드 수:").grid(row=0, column=1, padx=5, pady=2)
        self.nodes_var = tk.StringVar(value="64")
        self.nodes_entry = ttk.Entry(self, textvariable=self.nodes_var, width=8)
        self.nodes_entry.grid(row=0, column=2, padx=5, pady=2)

        # 활성화 함수 설정
        ttk.Label(self, text="활성화 함수:").grid(row=0, column=3, padx=5, pady=2)
        self.activation_var = tk.StringVar(value="relu")
        self.activation_combo = ttk.Combobox(self, textvariable=self.activation_var,
                                             values=["relu", "sigmoid", "tanh"], width=8)
        self.activation_combo.grid(row=0, column=4, padx=5, pady=2)

        # Dropout 설정
        ttk.Label(self, text="Dropout:").grid(row=0, column=5, padx=5, pady=2)
        self.dropout_var = tk.StringVar(value="0.2")
        self.dropout_entry = ttk.Entry(self, textvariable=self.dropout_var, width=8)
        self.dropout_entry.grid(row=0, column=6, padx=5, pady=2)

        # BatchNorm 설정
        self.batch_norm_var = tk.BooleanVar(value=True)
        self.batch_norm_check = ttk.Checkbutton(self, text="BatchNorm",
                                                variable=self.batch_norm_var)
        self.batch_norm_check.grid(row=0, column=7, padx=5, pady=2)

    def get_config(self):
        return {
            'nodes': int(self.nodes_var.get()),
            'activation': self.activation_var.get(),
            'dropout': float(self.dropout_var.get()),
            'batch_norm': self.batch_norm_var.get()
        }


class ModelTrainingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("신경망 모델 학습 프로그램")
        
        # 기본 창 크기 설정
        window_width = 1400
        window_height = 900
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        
        # 메인 프레임을 Canvas와 Scrollbar로 감싸기
        self.main_canvas = tk.Canvas(root)
        self.scrollbar = ttk.Scrollbar(root, orient="vertical", 
                                     command=self.main_canvas.yview)
        self.scrollable_frame = ttk.Frame(self.main_canvas)
        
        # 스크롤바 설정
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.main_canvas.configure(
                scrollregion=self.main_canvas.bbox("all")
            )
        )
        
        self.main_canvas.create_window((0, 0), window=self.scrollable_frame, 
                                     anchor="nw")
        self.main_canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # 스크롤바와 캔버스 배치
        self.main_canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        
        # root의 grid 가중치 설정
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)
        
        # 메인 프레임
        self.main_frame = ttk.Frame(self.scrollable_frame, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 데이터 선택 프레임 추가
        self.data_frame = ttk.LabelFrame(self.main_frame, text="데이터 설정", padding="5")
        self.data_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # 기존의 param_frame을 row=1로 이동
        self.param_frame = ttk.LabelFrame(self.main_frame, text="기본 파라미터", padding="5")
        self.param_frame.grid(row=1, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))

        # 에폭 설정
        ttk.Label(self.param_frame, text="에폭:").grid(row=0, column=0, padx=5, pady=5)
        self.epoch_var = tk.StringVar(value="100")  # 최대 에폭 수 증가
        self.epoch_entry = ttk.Entry(self.param_frame, textvariable=self.epoch_var, width=10)
        self.epoch_entry.grid(row=0, column=1, padx=5, pady=5)

        # 배치 크기 설정
        ttk.Label(self.param_frame, text="배치 크기:").grid(row=0, column=2, padx=5, pady=5)
        self.batch_var = tk.StringVar(value="16")
        self.batch_entry = ttk.Entry(self.param_frame, textvariable=self.batch_var, width=10)
        self.batch_entry.grid(row=0, column=3, padx=5, pady=5)

        # Early Stopping 설정
        ttk.Label(self.param_frame, text="Early Stopping:").grid(row=1, column=0, padx=5, pady=5)
        self.early_stopping_var = tk.BooleanVar(value=True)
        self.early_stopping_check = ttk.Checkbutton(self.param_frame,
                                                  text="사용",
                                                  variable=self.early_stopping_var)
        self.early_stopping_check.grid(row=1, column=1, padx=5, pady=5)

        # Patience 설정
        ttk.Label(self.param_frame, text="Patience:").grid(row=1, column=2, padx=5, pady=5)
        self.patience_var = tk.StringVar(value="5")
        self.patience_entry = ttk.Entry(self.param_frame, textvariable=self.patience_var, width=10)
        self.patience_entry.grid(row=1, column=3, padx=5, pady=5)

        # Monitor 설정
        ttk.Label(self.param_frame, text="Monitor:").grid(row=2, column=0, padx=5, pady=5)
        self.monitor_var = tk.StringVar(value="val_loss")
        self.monitor_combo = ttk.Combobox(self.param_frame,
                                        textvariable=self.monitor_var,
                                        values=["val_loss", "val_accuracy",
                                               "loss", "accuracy"],
                                        width=15)
        self.monitor_combo.grid(row=2, column=1, padx=5, pady=5)

        # Min Delta 설정
        ttk.Label(self.param_frame, text="Min Delta:").grid(row=2, column=2, padx=5, pady=5)
        self.min_delta_var = tk.StringVar(value="0.0001")
        self.min_delta_entry = ttk.Entry(self.param_frame,
                                       textvariable=self.min_delta_var, width=10)
        self.min_delta_entry.grid(row=2, column=3, padx=5, pady=5)

        # 기본 파라미터 프레임에 손실 함수 선택 옵션 추가
        ttk.Label(self.param_frame, text="손실 함수:").grid(row=3, column=0, padx=5, pady=5)
        self.loss_var = tk.StringVar(value="categorical_crossentropy")
        self.loss_combo = ttk.Combobox(self.param_frame,
                                     textvariable=self.loss_var,
                                     values=["categorical_crossentropy",
                                            "binary_crossentropy",
                                            "sparse_categorical_crossentropy",
                                            "mean_squared_error",
                                            "mean_absolute_error",
                                            "huber",
                                            "kullback_leibler_divergence"],
                                     width=25)
        self.loss_combo.grid(row=3, column=1, columnspan=2, padx=5, pady=5)

        # 나머지 프레임들의 row 번호도 1씩 증가
        self.layers_frame = ttk.LabelFrame(self.main_frame, text="레이어 설정", padding="5")
        self.layers_frame.grid(row=2, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        self.train_button = ttk.Button(self.main_frame, text="모델 학습 시작",
                                       command=self.train_model)
        self.train_button.grid(row=3, column=0, pady=10)
        self.result_frame = ttk.LabelFrame(self.main_frame, text="학습 결과",
                                         padding="5")
        self.result_frame.grid(row=4, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        self.graph_frame = ttk.Frame(self.main_frame)
        self.graph_frame.grid(row=5, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))

        # 데이터 소스 선택
        ttk.Label(self.data_frame, text="데이터 소스:").grid(row=0, column=0, padx=5, pady=5)
        self.data_source_var = tk.StringVar(value="iris")
        self.data_source_combo = ttk.Combobox(self.data_frame,
                                            textvariable=self.data_source_var,
                                            values=["iris", "csv_file"],
                                            width=15,
                                            state="readonly")
        self.data_source_combo.grid(row=0, column=1, padx=5, pady=5)
        self.data_source_combo.bind('<<ComboboxSelected>>', self.on_data_source_change)

        # 파일 선택 버튼 (초기에는 숨김)
        self.file_button = ttk.Button(self.data_frame, text="파일 선택",
                                    command=self.select_file)
        self.file_path_var = tk.StringVar()
        self.file_path_label = ttk.Label(self.data_frame,
                                       textvariable=self.file_path_var,
                                       wraplength=300)

        # 타겟 컬럼 선택
        ttk.Label(self.data_frame, text="타겟 컬럼:").grid(row=1, column=0, padx=5, pady=5)
        self.target_var = tk.StringVar()
        self.target_combo = ttk.Combobox(self.data_frame,
                                       textvariable=self.target_var,
                                       width=15,
                                       state="disabled")
        self.target_combo.grid(row=1, column=1, padx=5, pady=5)

        # 특성 선택
        ttk.Label(self.data_frame, text="사용할 특성:").grid(row=2, column=0, padx=5, pady=5)
        self.features_frame = ttk.Frame(self.data_frame)
        self.features_frame.grid(row=2, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        self.feature_vars = []

        # 레이어 설정 프레임
        self.layers_frame = ttk.LabelFrame(self.main_frame, text="레이어 설정", padding="5")
        self.layers_frame.grid(row=2, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))

        # 레이어 제어 버튼 프레임
        self.layer_control_frame = ttk.Frame(self.layers_frame)
        self.layer_control_frame.grid(row=0, column=0, columnspan=3, pady=5)

        # 레이어 추가/삭제 버튼
        self.add_layer_button = ttk.Button(self.layer_control_frame,
                                         text="레이어 추가",
                                         command=self.add_layer)
        self.add_layer_button.grid(row=0, column=0, padx=5)

        self.remove_layer_button = ttk.Button(self.layer_control_frame,
                                            text="마지막 레이어 삭제",
                                            command=self.remove_layer)
        self.remove_layer_button.grid(row=0, column=1, padx=5)

        # 레이어 설정을 위한 스크롤 캔버스 (높이 제한)
        self.layers_canvas = tk.Canvas(self.layers_frame, height=120)
        self.layers_scrollbar = ttk.Scrollbar(self.layers_frame,
                                            orient="vertical",
                                            command=self.layers_canvas.yview)
        self.layers_container = ttk.Frame(self.layers_canvas)

        self.layers_container.bind(
            "<Configure>",
            lambda e: self.layers_canvas.configure(
                scrollregion=self.layers_canvas.bbox("all")
            )
        )

        self.layers_canvas.create_window((0, 0), window=self.layers_container,
                                       anchor="nw")
        self.layers_canvas.configure(yscrollcommand=self.layers_scrollbar.set)

        self.layers_canvas.grid(row=1, column=0, columnspan=3, sticky="nsew")
        self.layers_scrollbar.grid(row=1, column=3, sticky="ns")

        # layers_frame의 grid 가중치 설정
        self.layers_frame.grid_rowconfigure(1, weight=1)
        self.layers_frame.grid_columnconfigure(0, weight=1)

        # 레이어 프레임의 크기 제한
        self.layers_frame.grid_propagate(False)  # 크기 고정
        self.layers_frame.configure(height=150)  # 전체 높이 제한

        self.layer_frames = []
        # 초기 레이어 하나 추가
        self.add_layer()

        # 학습 버튼
        self.train_button = ttk.Button(self.main_frame, text="모델 학습 시작",
                                       command=self.train_model)
        self.train_button.grid(row=3, column=0, pady=10)

        # 결과 표시 영역 (스크롤바 추가)
        self.result_frame = ttk.LabelFrame(self.main_frame, text="학습 결과",
                                         padding="5")
        self.result_frame.grid(row=4, column=0, padx=5, pady=5,
                             sticky=(tk.W, tk.E))

        # 텍스트 위젯과 스크롤바
        self.log_text = tk.Text(self.result_frame, height=10, width=50)
        self.log_scrollbar = ttk.Scrollbar(self.result_frame, orient="vertical",
                                         command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=self.log_scrollbar.set)

        self.log_text.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.log_scrollbar.grid(row=0, column=1, sticky="ns")

        # result_frame의 grid 가중치 설정
        self.result_frame.grid_rowconfigure(0, weight=1)
        self.result_frame.grid_columnconfigure(0, weight=1)

        # 그래프를 표시할 프레임
        self.graph_frame = ttk.Frame(self.main_frame)
        self.graph_frame.grid(row=5, column=0, padx=5, pady=5,
                            sticky=(tk.W, tk.E))

        # 그래프 크기 수정 - figsize를 더 작게 설정
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(8, 3))

        # 그래프 여백 조정
        self.fig.tight_layout(pad=1.5)  # 여백 추가

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # 마우스 휠 스크롤 이벤트 바인딩
        self.bind_mouse_wheel()

        # 현재 모델 저장
        self.current_model = None
        self.scaler = None

        ttk.Label(self.param_frame, text="문제 유형:").grid(row=3, column=0, padx=5, pady=5)
        self.problem_type_var = tk.StringVar(value="classification")
        self.problem_type_combo = ttk.Combobox(self.param_frame,
                                               textvariable=self.problem_type_var,
                                               values=["classification", "regression"],
                                               width=15,
                                               state="readonly")
        self.problem_type_combo.grid(row=3, column=1, padx=5, pady=5)
        self.problem_type_combo.bind('<<ComboboxSelected>>', self.on_problem_type_change)

        # 출력층 활성화 함수 선택 추가
        ttk.Label(self.param_frame, text="출력층 활성화 함수:").grid(row=4, column=0, padx=5, pady=5)
        self.output_activation_var = tk.StringVar(value="softmax")
        self.output_activation_combo = ttk.Combobox(self.param_frame,
                                                    textvariable=self.output_activation_var,
                                                    values=["softmax", "sigmoid", "linear"],
                                                    width=15)
        self.output_activation_combo.grid(row=4, column=1, padx=5, pady=5)

    def on_problem_type_change(self, event=None):
        """문제 유형이 변경될 때 UI 업데이트"""
        if self.problem_type_var.get() == "classification":
            self.loss_combo['values'] = ["categorical_crossentropy",
                                         "binary_crossentropy",
                                         "sparse_categorical_crossentropy"]
            self.loss_var.set("categorical_crossentropy")
            self.output_activation_var.set("softmax")
            self.output_activation_combo['values'] = ["softmax", "sigmoid"]
        else:  # regression
            self.loss_combo['values'] = ["mean_squared_error",
                                         "mean_absolute_error",
                                         "huber"]
            self.loss_var.set("mean_squared_error")
            self.output_activation_var.set("linear")
            self.output_activation_combo['values'] = ["linear", "relu", "tanh"]

    def on_data_source_change(self, event=None):
        """데이터 소스가 변경될 때 호출되는 함수"""
        if self.data_source_var.get() == "csv_file":
            self.file_button.grid(row=0, column=2, padx=5, pady=5)
            self.file_path_label.grid(row=0, column=3, padx=5, pady=5)
            self.target_combo.configure(state="readonly")
        else:
            self.file_button.grid_remove()
            self.file_path_label.grid_remove()
            self.file_path_var.set("")
            self.target_combo.configure(state="disabled")
            self.target_var.set("")
            self.clear_feature_checkboxes()

    def select_file(self):
        """CSV 파일 선택 다이얼로그를 표시"""
        from tkinter import filedialog
        import pandas as pd

        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if file_path:
            try:
                # CSV 파일 읽기
                df = pd.read_csv(file_path)
                self.file_path_var.set(file_path)

                # 컬럼 목록 업데이트
                columns = df.columns.tolist()
                self.target_combo['values'] = columns
                self.target_combo.set(columns[-1])  # 마지막 컬럼을 기본값으로

                # 특성 체크박스 업데이트
                self.update_feature_checkboxes(columns)

            except Exception as e:
                messagebox.showerror("에러", f"파일을 읽는 중 오류가 발생했습니다:\n{str(e)}")

    def clear_feature_checkboxes(self):
        """특성 체크박스들을 모두 제거"""
        for widget in self.features_frame.winfo_children():
            widget.destroy()
        self.feature_vars.clear()

    def update_feature_checkboxes(self, columns):
        """특성 선택을 위한 체크박스 업데이트"""
        self.clear_feature_checkboxes()

        for i, col in enumerate(columns):
            var = tk.BooleanVar(value=True)
            self.feature_vars.append(var)
            cb = ttk.Checkbutton(self.features_frame, text=col, variable=var)
            cb.grid(row=i//3, column=i%3, sticky='w', padx=5)

    def bind_mouse_wheel(self):
        """마우스 휠 이벤트를 모든 관련 위젯에 바인딩"""
        def _on_mousewheel(event):
            if event.delta:
                delta = event.delta
            else:
                if event.num == 5:
                    delta = -1
                else:
                    delta = 1
            self.main_canvas.yview_scroll(int(-1*(delta/120)), "units")

        def _on_mousewheel_layers(event):
            if event.delta:
                delta = event.delta
            else:
                if event.num == 5:
                    delta = -1
                else:
                    delta = 1
            self.layers_canvas.yview_scroll(int(-1*(delta/120)), "units")

        # Windows의 경우
        self.main_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.layers_canvas.bind_all("<MouseWheel>", _on_mousewheel_layers)

        # Linux의 경우
        self.main_canvas.bind_all("<Button-4>", _on_mousewheel)
        self.main_canvas.bind_all("<Button-5>", _on_mousewheel)
        self.layers_canvas.bind_all("<Button-4>", _on_mousewheel_layers)
        self.layers_canvas.bind_all("<Button-5>", _on_mousewheel_layers)

    def add_layer(self):
        """새로운 레이어를 추가합니다."""
        layer_num = len(self.layer_frames) + 1
        layer_frame = LayerFrame(self.layers_container, layer_num)
        layer_frame.grid(row=layer_num-1, column=0, padx=5, pady=2, sticky=(tk.W, tk.E))
        self.layer_frames.append(layer_frame)

        # 스크롤 영역 업데이트
        self.layers_canvas.configure(scrollregion=self.layers_canvas.bbox("all"))

    def remove_layer(self):
        """마지막 레이어를 삭제합니다."""
        if len(self.layer_frames) > 1:  # 최소 1개의 레이어는 유지
            layer_frame = self.layer_frames.pop()
            layer_frame.destroy()

            # 나머지 레이어들의 번호 업데이트
            for i, frame in enumerate(self.layer_frames, 1):
                frame.layer_num = i
                frame.grid(row=i-1, column=0)
                frame.grid_remove()
                frame.grid()

            # 스크롤 영역 업데이트
            self.layers_canvas.configure(scrollregion=self.layers_canvas.bbox("all"))
        else:
            messagebox.showwarning("경고", "최소 1개의 레이어는 유지해야 합니다.")

    def load_and_preprocess_data(self):
        """데이터 로드 및 전처리"""
        if self.data_source_var.get() == "iris":
            # 기존 iris 데이터셋 로드
            iris = load_iris()
            X = iris.data
            y = iris.target
            y = tf.keras.utils.to_categorical(y)
        else:
            # CSV 파일에서 데이터 로드

            if not self.file_path_var.get():
                raise ValueError("CSV 파일을 선택해주세요.")

            df = pd.read_csv(self.file_path_var.get())

            # 선택된 특성과 타겟 분리
            selected_features = [col for col, var in zip(df.columns, self.feature_vars)
                              if var.get() and col != self.target_var.get()]

            if not selected_features:
                raise ValueError("최소 하나 이상의 특성을 선택해주세요.")

            X = df[selected_features].values
            y = df[self.target_var.get()].values

            # 타겟 데이터 원-핫 인코딩
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)
            y = tf.keras.utils.to_categorical(y)

        # 데이터 분할 및 스케일링
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.y_train = y_train

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def create_model(self, layer_configs):
        input_dim = len([var for var in self.feature_vars if var.get()]) if self.data_source_var.get() != "iris" else 4

        if self.problem_type_var.get() == "classification":
            if self.data_source_var.get() == "iris":
                output_dim = 3
            else:
                output_dim = len(np.unique(np.argmax(self.y_train, axis=1)))
        else:  # regression
            output_dim = 1  # 회귀의 경우 출력 차원은 1

        model = Sequential()
        model.add(Input((input_dim,)))

        for config in layer_configs:
            model.add(Dense(config['nodes'], activation=config['activation']))
            if config['batch_norm']:
                model.add(BatchNormalization())
            if config['dropout'] > 0:
                model.add(Dropout(config['dropout']))

        model.add(Dense(output_dim, activation=self.output_activation_var.get()))

        metrics = ['accuracy'] if self.problem_type_var.get() == "classification" else ['mae', 'mse']

        model.compile(
            optimizer='adam',
            loss=self.loss_var.get(),
            metrics=metrics
        )

        return model


    def update_log(self, text):
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)
        self.root.update()

    def plot_history(self, history):
        self.ax1.clear()
        self.ax2.clear()
        font_size = 8

        # 손실 그래프
        self.ax1.plot(history.history['loss'], label='train')
        self.ax1.plot(history.history['val_loss'], label='validation')
        self.ax1.set_title('Model Loss', fontsize=font_size + 1)
        self.ax1.set_xlabel('Epoch', fontsize=font_size)
        self.ax1.set_ylabel('Loss', fontsize=font_size)
        self.ax1.tick_params(labelsize=font_size - 1)
        self.ax1.legend(fontsize=font_size - 1)

        # 메트릭 그래프
        if self.problem_type_var.get() == "classification":
            metric = 'accuracy'
            title = 'Model Accuracy'
        else:
            metric = 'mae'  # mean absolute error
            title = 'Model MAE'

        self.ax2.plot(history.history[metric], label='train')
        self.ax2.plot(history.history[f'val_{metric}'], label='validation')
        self.ax2.set_title(title, fontsize=font_size + 1)
        self.ax2.set_xlabel('Epoch', fontsize=font_size)
        self.ax2.set_ylabel(metric.upper(), fontsize=font_size)
        self.ax2.tick_params(labelsize=font_size - 1)
        self.ax2.legend(fontsize=font_size - 1)

        self.fig.tight_layout()
        self.canvas.draw()


    def train_model(self):
        try:
            # 파라미터 가져오기
            epochs = int(self.epoch_var.get())
            batch_size = int(self.batch_var.get())

            # Early Stopping 설정 가져오기
            callbacks = []
            early_stopping_used = False
            if self.early_stopping_var.get():
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor=self.monitor_var.get(),
                    patience=int(self.patience_var.get()),
                    min_delta=float(self.min_delta_var.get()),
                    restore_best_weights=True,
                    verbose=1  # Early Stopping 메시지 표시
                )
                callbacks.append(early_stopping)
                early_stopping_used = True

            # 레이어 설정 가져오기
            layer_configs = [frame.get_config() for frame in self.layer_frames]

            # 로그 초기화
            self.log_text.delete(1.0, tk.END)

            # 데이터 로드
            self.update_log("데이터 로드 중...")
            X_train_scaled, X_test_scaled, y_train, y_test = self.load_and_preprocess_data()

            # 모델 생성
            self.update_log("모델 생성 중...")
            model = self.create_model(layer_configs)

            # 모델 구조 출력
            self.update_log("\n모델 구조:")
            model.summary(print_fn=lambda x: self.update_log(x))

            # 콜백 설정
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )

            # 모델 학습
            self.update_log("\n학습 시작...")
            history = model.fit(
                X_train_scaled,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1
            )

            # Early Stopping 정보 출력
            if early_stopping_used:
                stopped_epoch = len(history.history['loss'])
                if stopped_epoch < epochs:  # Early Stopping이 발생한 경우
                    self.update_log(f"전체 {epochs}개의 에폭 중 {stopped_epoch}개의 에폭만 실행되었습니다.")
                    best_epoch = np.argmin(history.history[self.monitor_var.get()])
                    self.update_log(f"최적의 {self.monitor_var.get()} 값은 {best_epoch + 1}번째 에폭에서 달성되었습니다.")

            # 테스트 세트 평가
            test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
            self.update_log(f'\n테스트 세트 손실: {test_loss:.4f}')
            self.update_log(f'테스트 세트 정확도: {test_acc:.4f}')

            # 그래프 그리기
            self.plot_history(history)



        except Exception as e:
            messagebox.showerror("에러", str(e))
            self.update_log(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ModelTrainingGUI(root)
    root.mainloop()