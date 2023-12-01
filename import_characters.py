import sqlite3
import yaml
import glob

con=sqlite3.connect("ltm/memories.db")
cursor = con.cursor()
cursor.execute("SELECT character FROM personas")
existing = cursor.fetchall()
print(existing)
for persona in glob.glob("personas/*.yaml"):
    print(persona)
    try:
        with open(persona) as f:
            p = yaml.safe_load(f)
            print(p['name'])
            sql = ""
            if p["name"] not in existing:
                sql = f'INSERT INTO personas (character, description) VALUES("{p["name"]}", "{p["context"]}")'
            else:
                sql = f'UPDATE personas SET description="{p["context"]}" WHERE character={p["name"]}'
            print(sql)
            cursor.execute(sql)
            con.commit()
    except:
        continue
con.close()
