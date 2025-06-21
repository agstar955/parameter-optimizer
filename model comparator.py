import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import tensorflow as tf
import numpy as np
import os


class ModelComparisonGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("딥러닝 모델 비교 프로그램")

        # 기본 창 크기 설정
        window_width = 800
        window_height = 600
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        center_x = int(screen_width / 2 - window_width / 2)
        center_y = int(screen_height / 2 - window_height / 2)
        root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

        # 메인 프레임
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 모델 목록 프레임
        self.models_frame = ttk.LabelFrame(self.main_frame, text="모델 목록", padding="5")
        self.models_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 모델 목록 트리뷰
        self.tree = ttk.Treeview(self.models_frame,
                                 columns=("모델명", "문제유형", "정확도/MAE", "손실값", "에폭", "배치크기"),
                                 show="headings")
        self.tree.heading("모델명", text="모델명")
        self.tree.heading("문제유형", text="문제유형")
        self.tree.heading("정확도/MAE", text="정확도/MAE")
        self.tree.heading("손실값", text="손실값")
        self.tree.heading("에폭", text="에폭")
        self.tree.heading("배치크기", text="배치크기")

        self.tree.column("모델명", width=150)
        self.tree.column("문제유형", width=100)
        self.tree.column("정확도/MAE", width=100)
        self.tree.column("손실값", width=100)
        self.tree.column("에폭", width=70)
        self.tree.column("배치크기", width=70)

        self.tree.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 스크롤바 추가
        scrollbar = ttk.Scrollbar(self.models_frame, orient="vertical", command=self.tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scrollbar.set)

        # 버튼 프레임
        self.button_frame = ttk.Frame(self.models_frame)
        self.button_frame.grid(row=1, column=0, columnspan=2, pady=5)

        # 모델 불러오기 버튼
        self.load_button = ttk.Button(self.button_frame, text="모델 추가", command=self.load_model)
        self.load_button.grid(row=0, column=0, padx=5)

        # 선택 모델 삭제 버튼
        self.delete_button = ttk.Button(self.button_frame, text="선택 모델 삭제", command=self.delete_model)
        self.delete_button.grid(row=0, column=1, padx=5)

        # 모델 세부 정보 프레임
        self.detail_frame = ttk.LabelFrame(self.main_frame, text="모델 상세 정보", padding="5")
        self.detail_frame.grid(row=1, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))

        # 상세 정보 텍스트 위젯
        self.detail_text = tk.Text(self.detail_frame, height=15, width=80)
        self.detail_text.grid(row=0, column=0, padx=5, pady=5)

        # 스크롤바 추가
        detail_scrollbar = ttk.Scrollbar(self.detail_frame, orient="vertical",
                                         command=self.detail_text.yview)
        detail_scrollbar.grid(row=0, column=1, sticky="ns")
        self.detail_text.configure(yscrollcommand=detail_scrollbar.set)

        # 모델 저장소
        self.models = {}
        self.configs = {}

        # 트리뷰 선택 이벤트 바인딩
        self.tree.bind('<<TreeviewSelect>>', self.show_model_details)

        # 그리드 설정
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=3)
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.models_frame.grid_rowconfigure(0, weight=1)
        self.models_frame.grid_columnconfigure(0, weight=1)

    def load_model(self):
        """모델 파일 로드"""
        file_paths = filedialog.askopenfilenames(
            filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
        )

        for file_path in file_paths:
            model_name = os.path.basename(file_path)
            try:
                config_path = file_path.replace('.h5', '_config.json')

                if not os.path.exists(config_path):
                    messagebox.showwarning("경고", f"{model_name}의 설정 파일이 없습니다.")
                    continue

                # 모델과 설정 로드
                model = tf.keras.models.load_model(file_path)
                with open(config_path, 'r') as f:
                    config = json.load(f)

                # 모델 저장
                self.models[model_name] = model
                self.configs[model_name] = config

                # 트리뷰에 추가
                problem_type = config['problem_type']
                metric = "정확도" if problem_type == "classification" else "MAE"

                self.tree.insert("", "end", values=(
                    model_name,
                    problem_type,
                    f"{config['accuracy'][1]:.4f}",
                    f"{config['accuracy'][0]:.4f}",
                    config['epochs'],
                    config['batch_size']
                ))

            except Exception as e:
                messagebox.showerror("에러", f"{model_name} 로드 중 오류 발생:\n{str(e)}")

    def delete_model(self):
        """선택한 모델 삭제"""
        selected_items = self.tree.selection()
        if not selected_items:
            messagebox.showwarning("경고", "삭제할 모델을 선택해주세요.")
            return

        for item in selected_items:
            model_name = self.tree.item(item)['values'][0]
            del self.models[model_name]
            del self.configs[model_name]
            self.tree.delete(item)

    def show_model_details(self, event):
        """선택한 모델의 상세 정보 표시"""
        selected_items = self.tree.selection()
        if not selected_items:
            return

        self.detail_text.delete(1.0, tk.END)

        for item in selected_items:
            model_name = self.tree.item(item)['values'][0]
            config = self.configs[model_name]
            model = self.models[model_name]

            details = f"모델명: {model_name}\n"
            details += f"문제 유형: {config['problem_type']}\n"
            details += f"손실 함수: {config['loss_function']}\n"
            details += f"에폭: {config['epochs']}\n"
            details += f"배치 크기: {config['batch_size']}\n"
            details += f"출력층 활성화 함수: {config['output_activation']}\n\n"

            details += "레이어 구성:\n"
            for i, layer in enumerate(config['layers'], 1):
                details += f"레이어 {i}:\n"
                details += f"  - 노드 수: {layer['nodes']}\n"
                details += f"  - 활성화 함수: {layer['activation']}\n"
                details += f"  - Dropout: {layer['dropout']}\n"
                details += f"  - BatchNorm: {layer['batch_norm']}\n"

            details+=f"\n정확도/MAE: {config['accuracy'][1]:.4f}\n"
            details+=f"손실: {config['accuracy'][0]:.4f}\n"

            self.detail_text.insert(tk.END, details + "\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = ModelComparisonGUI(root)
    root.mainloop()