class DataType {
    int age;
    double height;
    char grade;
    boolean pass;
    String name;

    void display() {
        System.out.println(age);
        System.out.println(height);
        System.out.println(grade);
        System.out.println(pass);
        System.out.println(name);
    }

    public static void main(String[] args) {
        DataType dt = new DataType();
        dt.age = 20;
        dt.height = 5.9;
        dt.grade = 'A';
        dt.pass = true;
        dt.name = "Niku";
        dt.display();
    }
}