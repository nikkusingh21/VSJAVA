class Construct {
   int roll;
   String name;

   
   Construct() {
       roll = 24;
       name = "Null"; 
   }

   void show() {
       System.out.println(roll + " " + name);
   }

   public static void main(String[] args) {
       Construct ref = new Construct(); 
       ref.show();                      
   }
}