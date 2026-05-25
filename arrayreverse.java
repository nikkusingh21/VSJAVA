class arrayreverse {
    public static void main(String[] args) {

        int[] arr = {10, 20, 30, 40, 50};

        System.out.println("Reversed Array:");

        for(int i = arr.length - 1; i >= 0; i--) {
            System.out.println(arr[i]);
        }
    }
}