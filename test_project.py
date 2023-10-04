import project
from project import df_nofire,df_fire, image_to_numerical_rgb
def main():
    test_project()
def test_df_nofire():
    assert df_nofire("demo_2.jpg").iloc[0,3]==2
    assert df_nofire("river.jpg").iloc[0,3]==2
def test_nofire():
    assert df_fire("fire.jfif").iloc[0,3]==1
    assert df_fire("fire_2.jpg").iloc[0,3]==1
def test_image_to_numerical_rgb():
    assert image_to_numerical_rgb("demo_2.jpg")[0][0]==96
    assert image_to_numerical_rgb("demo_2.jpg")[0][1]==133
    assert image_to_numerical_rgb("demo_2.jpg")[0][2]==63

    assert image_to_numerical_rgb("fire.jfif")[0][0]==1
    assert image_to_numerical_rgb("fire.jfif")[0][1]==1
    assert image_to_numerical_rgb("fire.jfif")[0][2]==1


if __name__ == "__main__":
    main()