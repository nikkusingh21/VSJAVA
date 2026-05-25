class AvgArray {
    public static void main(String[] args) {
        int[] arr = {20, 40, 60, 80};

        int sum = 0;
        for (int n : arr) {
            sum += n;
        }

        double avg = (double) sum / arr.length;
        System.out.println("Average = " + avg);
    }
}