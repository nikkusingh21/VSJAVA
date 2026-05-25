import java.util.Scanner;

class Geeks {
    public static void main(String[] args)
    {
        // Creating Scanner object
        Scanner sc = new Scanner(System.in);

        System.out.print("Enter First Number: ");
        int a = sc.nextInt();

        System.out.print("Enter Second Number: ");
        int b = sc.nextInt();

        System.out.println("Sum: " + (a + b));
        sc.close();
    }
}