import java.util.Scanner;

class IsPalindrome {

    public static void main(String[] args) {

        Scanner sc = new Scanner(System.in);

        System.out.print("Enter string: ");
        String s = sc.nextLine();

        int left = 0, right = s.length() - 1;
        boolean palindrome = true;

        while (left < right) {

            while (left < right && 
                  !Character.isLetterOrDigit(s.charAt(left))) {
                left++;
            }

            while (left < right && 
                  !Character.isLetterOrDigit(s.charAt(right))) {
                right--;
            }

            if (Character.toLowerCase(s.charAt(left)) !=
                Character.toLowerCase(s.charAt(right))) {

                palindrome = false;
                break;
            }

            left++;
            right--;
        }

        if (palindrome)
            System.out.println("Palindrome");
        else
            System.out.println("Not Palindrome");

        sc.close();
    }
}