import java.util.Scanner;

public class BitwiseOperator {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        System.out.print("Enter first number: ");
        int a = sc.nextInt();
        System.out.print("Enter second number: ");
        int b = sc.nextInt();

        System.out.println("a & b (AND): " + (a & b));
        System.out.println("a | b (OR): " + (a | b));
        System.out.println("a ^ b (XOR): " + (a ^ b));
        System.out.println("~a (NOT): " + (~a));
        System.out.println("a << 1 (Left Shift): " + (a << 1));
        System.out.println("a >> 1 (Right Shift): " + (a >> 1));

        sc.close();
    }
}