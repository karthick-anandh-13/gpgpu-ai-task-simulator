# simulator.py
import os
from tasks import matmul_task, relu_task, sigmoid_task, softmax_task, tanh_task

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    while True:
        clear_console()
        print("\n" + "="*50)
        print("🧠 GPGPU AI Task Simulator")
        print("="*50)
        print("1. 🧮 Matrix Multiplication")
        print("2. 🔋 ReLU Activation")
        print("3. 🔄 Sigmoid Activation")
        print("4. 📊 Softmax Activation")
        print("5. 🌊 Tanh Activation")
        print("0. ❌ Exit")
        print("="*50)

        choice = input("👉 Select a task: ")

        if choice == "1":
            matmul_task.run_matmul()
        elif choice == "2":
            relu_task.run_relu()
        elif choice == "3":
            sigmoid_task.run_sigmoid()
        elif choice == "4":
            softmax_task.run_softmax()
        elif choice == "5":
            tanh_task.run_tanh()
        elif choice == "0":
            print("👋 Exiting... Have a great day!")
            break
        else:
            print("❌ Invalid option. Please try again.")
        input("\n🔁 Press Enter to return to the menu...")

if __name__ == "__main__":
    main()