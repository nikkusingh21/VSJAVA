class MoveZeroes {
    public static void main(String[] args) {
        int[] arr = {0, 1, 0, 15, 2};

        int index = 0;

        for (int arr1 : arr) {
            if (arr1 != 0) {
                arr[index] = arr1;
                index++;
            }
        }

        while (index < arr.length) {
            arr[index] = 0;
            index++;
        }

        for (int n : arr) {
            System.out.print(n + " ");
        }
    }
}