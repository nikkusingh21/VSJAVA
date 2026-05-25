class Car {
    String color;
    void drive() {
        System.out.println("Car is driving");
    }
}

class SportsCar extends Car {
    void turbo() {
        System.out.println("Turbo mode activated!");
    }
}

class InheritanceDemo {
    public static void main(String[] args) {
        SportsCar sc = new SportsCar();
        sc.color = "Blue";
        sc.drive();
        sc.turbo();
    }
}