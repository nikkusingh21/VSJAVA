import java.util.Scanner;

class ArithmeticOperations {
    int add(int a, int b) { return a + b; }
    int subtract(int a, int b) { return a - b; }
    int multiply(int a, int b) { return a * b; }
    String divide(int a, int b) {
        return (b != 0) ? String.valueOf(a / b) : "Cannot divide by zero";
    }
    String modulus(int a, int b) {
        return (b != 0) ? String.valueOf(a % b) : "Cannot divide by zero";
    }
}

public class Arithmeticoperator {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        ArithmeticOperations op = new ArithmeticOperations();

        System.out.print("Enter first number: ");
        int a = sc.nextInt();
        System.out.print("Enter second number: ");
        int b = sc.nextInt();

        System.out.println("Addition: " + op.add(a, b));
        System.out.println("Subtraction: " + op.subtract(a, b));
        System.out.println("Multiplication: " + op.multiply(a, b));
        System.out.println("Division: " + op.divide(a, b));
        System.out.println("Modulus: " + op.modulus(a, b));

        sc.close();
    }
}