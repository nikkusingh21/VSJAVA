//sum of array{10,5,15,50}
public class sumarray {
   
    public static void main(String args[])
    {
        int arr[] = { 10,5,15,50 };

        
        sum(arr);
    }

    public static void sum(int[] arr)
    {
        
        int sum = 0;

        for (int i = 0; i < arr.length; i++)
            sum += arr[i];

        System.out.println("sum of array values : " + sum);
    }
}