import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
from tasks import relu_task, sigmoid_task, softmax_task, tanh_task, matmul_task

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class GPGPUApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("🚀 GPGPU AI Task Simulator")
        self.geometry("600x520")
        self.resizable(False, False)

        self.title_label = ctk.CTkLabel(self, text="Select and Run AI Task", font=("Precious", 24))
        self.title_label.pack(pady=20)

        self.task_dropdown = ctk.CTkOptionMenu(
            self,
            values=["Matrix Multiplication", "ReLU", "Sigmoid", "Softmax", "Tanh"]
        )
        self.task_dropdown.pack(pady=10)

        self.run_button = ctk.CTkButton(self, text="▶ Run Task", command=self.run_selected_task)
        self.run_button.pack(pady=15)

        self.file_info = ctk.CTkLabel(self, text="Uses CSV input/output via file dialog", font=("Arial", 12))
        self.file_info.pack(pady=5)

        self.status_box = ctk.CTkTextbox(self, height=220, width=520)
        self.status_box.pack(pady=10)
        self.status_box.insert("0.0", "Status: Waiting for task selection...\n")
        self.status_box.configure(state="disabled")

    def update_status(self, msg):
        self.status_box.configure(state="normal")
        self.status_box.insert(ctk.END, msg + "\n")
        self.status_box.configure(state="disabled")
        self.status_box.see(ctk.END)

    def choose_input_output_files(self, task_name):
        input_file = filedialog.askopenfilename(title=f"Select input CSV for {task_name}", filetypes=[("CSV files", "*.csv")])
        if not input_file:
            self.update_status("❌ Input file not selected.")
            return None, None
        self.update_status(f"📂 Input file loaded: {input_file}")

        output_file = filedialog.asksaveasfilename(title=f"Save output of {task_name} as", defaultextension=".csv")
        if not output_file:
            self.update_status("❌ Output file not selected.")
            return None, None
        self.update_status(f"💾 Output will be saved to: {output_file}")

        return input_file, output_file

    def run_selected_task(self):
        task_name = self.task_dropdown.get()
        self.update_status(f"\n⚙️ Running {task_name}...")

        try:
            if task_name == "Matrix Multiplication":
                file_a = filedialog.askopenfilename(title="Select Matrix A CSV", filetypes=[("CSV files", "*.csv")])
                if not file_a:
                    self.update_status("❌ Matrix A file not selected.")
                    return
                self.update_status(f"📂 Matrix A loaded: {file_a}")

                file_b = filedialog.askopenfilename(title="Select Matrix B CSV", filetypes=[("CSV files", "*.csv")])
                if not file_b:
                    self.update_status("❌ Matrix B file not selected.")
                    return
                self.update_status(f"📂 Matrix B loaded: {file_b}")

                output_file = filedialog.asksaveasfilename(title="Save Result As", defaultextension=".csv")
                if not output_file:
                    self.update_status("❌ Output file not selected.")
                    return
                self.update_status(f"💾 Output will be saved to: {output_file}")

                matmul_task.run_matmul(file_a, file_b, output_file)

            elif task_name == "ReLU":
                input_file, output_file = self.choose_input_output_files("ReLU")
                if input_file and output_file:
                    relu_task.run_relu(input_file, output_file)

            elif task_name == "Sigmoid":
                input_file, output_file = self.choose_input_output_files("Sigmoid")
                if input_file and output_file:
                    sigmoid_task.run_sigmoid(input_file, output_file)

            elif task_name == "Softmax":
                input_file, output_file = self.choose_input_output_files("Softmax")
                if input_file and output_file:
                    softmax_task.run_softmax(input_file, output_file)

            elif task_name == "Tanh":
                input_file, output_file = self.choose_input_output_files("Tanh")
                if input_file and output_file:
                    tanh_task.run_tanh(input_file, output_file)

            else:
                self.update_status("❌ Invalid task selected.")
                return

            self.update_status(f"✅ {task_name} completed successfully!")

        except Exception as e:
            self.update_status(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    app = GPGPUApp()
    app.mainloop()
