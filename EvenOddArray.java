class EvenOddArray {
    public static void main(String[] args) {
        int[] arr = {10,20,15,1,2,3,6};
        int evenCount = 0;
        int oddCount = 0;

        for (int num : arr) {
            if (num % 2 == 0) {
                evenCount++;
            } else {
                oddCount++;
            }
        }

        int[] evenArray = new int[evenCount];
        int[] oddArray = new int[oddCount];
        int evenIndex = 0;
        int oddIndex = 0;

        for (int num : arr) {
            if (num % 2 == 0) {
                evenArray[evenIndex++] = num;
            } else {
                oddArray[oddIndex++] = num;
            }
        }

        System.out.println("Even numbers:");
        for (int num : evenArray) {
            System.out.print(num + " ");
        }
        System.out.println();

        System.out.println("Odd numbers:");
        for (int num : oddArray) {
            System.out.print(num + " ");
        }
    }
}