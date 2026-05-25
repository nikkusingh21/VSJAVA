import java.util.Scanner;

class CharCount {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter a string: ");
        String input = sc.nextLine();
        System.out.println("Character count: " + input.length());
        sc.close();
    }
}