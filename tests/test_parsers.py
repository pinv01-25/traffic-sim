"""Tests de parsers de outputs SUMO."""
import textwrap

from visualization.parsers import parse_summary, parse_tripinfo


def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(textwrap.dedent(content))
    return str(p)


def test_parse_tripinfo(tmp_path):
    path = _write(tmp_path, "tripinfo.xml", """\
        <?xml version="1.0" encoding="UTF-8"?>
        <tripinfos>
            <tripinfo id="car_1" depart="0.00" arrival="80.00" duration="80.00"
                      routeLength="500.0" departDelay="1.00" timeLoss="12.5"
                      waitingTime="4.0" departLane="e1_0"/>
            <tripinfo id="car_2" depart="5.00" arrival="95.00" duration="90.00"
                      routeLength="600.0" departDelay="0.00" timeLoss="20.0"
                      waitingTime="8.0" departLane="e2_0"/>
        </tripinfos>
    """)
    df = parse_tripinfo(path)
    assert len(df) == 2
    assert df.loc[df['id'] == 'car_1', 'duration'].iloc[0] == 80.0
    assert df.loc[df['id'] == 'car_2', 'timeLoss'].iloc[0] == 20.0


def test_parse_summary_step_format(tmp_path):
    # Formato real de SUMO: elementos <step>
    path = _write(tmp_path, "summary.xml", """\
        <?xml version="1.0" encoding="UTF-8"?>
        <summary>
            <step time="0.00" loaded="10" inserted="5" running="5" waiting="0"
                  halting="1" meanSpeed="8.5" meanWaitingTime="0.5"/>
            <step time="1.00" loaded="12" inserted="7" running="6" waiting="1"
                  halting="2" meanSpeed="7.9" meanWaitingTime="0.8"/>
        </summary>
    """)
    df = parse_summary(path)
    assert len(df) == 2
    assert list(df['running']) == [5.0, 6.0]
    assert list(df['halting']) == [1.0, 2.0]
    assert df['time'].iloc[1] == 1.0


def test_parse_summary_interval_format(tmp_path):
    path = _write(tmp_path, "summary.xml", """\
        <?xml version="1.0" encoding="UTF-8"?>
        <summary>
            <interval begin="0.00" end="60.00" running="4" halting="0"/>
        </summary>
    """)
    df = parse_summary(path)
    assert len(df) == 1
    assert df['running'].iloc[0] == 4.0
