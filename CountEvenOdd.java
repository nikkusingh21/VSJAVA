public class CountEvenOdd {
    public static void main(String[] args) {
        int arr[]={10,20,15,1,2,3,6};
        int evenCount=0;
        int oddCount=0;
        for(int i=0;i<arr.length;i++){
            if(arr[i]%2==0){
                evenCount++;
            }else{
                oddCount++;
            }
        }
        System.out.println("Even count: "+evenCount);
        System.out.println("Odd count: "+oddCount);
    }
}