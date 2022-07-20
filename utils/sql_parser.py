import time
import sqlparse
from sqlparse.tokens import Token
from utils.schema import Query, QueryTable, Schema, Table,TableAttribute

def _extract_identifiers(tokens, enforce_single=True):
    """
    extract identifiers from tokens
    :param tokens: parsed tokens
    :param enforce_single: whether to ensure only one identifiers
    :return: single identifier or list of identifier
    """
    identifiers = [token for token in tokens if isinstance(token, sqlparse.sql.IdentifierList)]
    if len(identifiers) >= 1:  # a list of identifiers
        if enforce_single:
            assert len(identifiers) == 1
        identifiers = identifiers[0]
    else:  # only one identifiers
        identifiers = [token for token in tokens if isinstance(token, sqlparse.sql.Identifier)]
    return identifiers


# Find corresponding table of attribute
def _find_matching_table(attribute, schema):
    """
    find the table to which the attribute belongs
    :param attribute: attribute name
    :param schema: schema information of tables
    :return: table name
    """
    table_name = None
    for table_obj in schema.tables:
        if attribute in table_obj.columns:
            table_name = table_obj.table_name
    assert table_name is not None, f"No table found for attribute {attribute}."
    return table_name


def _fully_qualified_attribute_name(identifier, schema, return_split=False):
    """
    find the full qualified name of attribute, like sales.price
    :param identifier: attribute identifier
    :param schema: schema information of tables
    :param return_split: the return form, Ture=>(table, attribute), False=>table.attribute
    :return: the full qualified name of attribute
    """
    if len(identifier.tokens) == 1:
        attribute = identifier.tokens[0].value
        table_name = _find_matching_table(attribute, schema)
        if not return_split:
            return table_name + '.' + attribute
        else:
            return table_name, attribute
    # If the identifier is already fully qualified
    assert identifier.tokens[1].value == '.', "Invalid Identifier"
    if not return_split:
        return identifier.tokens[0].value + '.' + identifier.tokens[2].value
    else:
        return identifier.tokens[0].value, identifier.tokens[2].value

def _extract_attribute(identifier, schema):
    if len(identifier.tokens) == 1:
        attr_name = identifier.tokens[0].value
        table_name = _find_matching_table(attr_name, schema)
        return TableAttribute(table_name,attr_name)
    return TableAttribute(identifier.tokens[0].value,identifier.tokens[2].value)


def _parse_aggregation(function, query, schema):
    operation_type = None
    operator = _extract_identifiers(function.tokens)[0]
    if operator.normalized == 'sum' or operator.normalized == 'SUM':
        operation_type = 'SUM'
    elif operator.normalized == 'avg' or operator.normalized == 'AVG':
        operation_type = 'AVG'
    elif operator.normalized == 'count' or operator.normalized == 'COUNT':
        query.add_aggregation_operation(('COUNT', None))
        return
    else:
        raise Exception(f"Unknown operator: {operator.normalized} ")
    operand_parantheses = [token for token in function if isinstance(token, sqlparse.sql.Parenthesis)]
    assert len(operand_parantheses) == 1
    operand_parantheses = operand_parantheses[0]
    operation_tokens = [token for token in operand_parantheses
                        if isinstance(token, sqlparse.sql.Operation)]
    identifier=_extract_identifiers(operand_parantheses)[0]
    attr=_extract_attribute(identifier,schema)
    # feature = _fully_qualified_attribute_name(_extract_identifiers(operand_parantheses)[0], schema, return_split=True)
    query.add_aggregation_operation((operation_type, attr))


def handle_aggregation(query, schema, tokens_before_from):
    operations = [token for token in tokens_before_from if isinstance(token, sqlparse.sql.Operation)]
    assert len(operations) == 0, "Operation is not supported currently."
    functions = [token for token in tokens_before_from if isinstance(token, sqlparse.sql.Function)]
    # assert len(functions) == 1, "Only a single aggregate function is supported."
    for function in functions:
        _parse_aggregation(function, query, schema)
    


def parse_query(query_str,spark):
    """
    Parses simple SQL queries and returns cardinality query object.
    :param query_str: sql query string
    :param schema: schema information
    :return:
    """
    schema = Schema()
    query = Query()
    # catalog=spark.catalog
    query_str=query_str.upper()
    # handle sampling method
    with_idx=query_str.rfind('WITH')
    with_clause=query_str[with_idx+5:]
    sampling_methods=with_clause.split(' ')
    query.multi_sampling_times=int(sampling_methods[0].split('=')[1])
    sampling_methods.pop(0)

    query_str=query_str[:with_idx]
    query.sql=query_str
    # split query into part before from
    parsed_tokens = sqlparse.parse(query_str)[0]
    from_idxs = [i for i, token in enumerate(parsed_tokens) if token.normalized == 'FROM']
    assert len(from_idxs) == 1, "Nested queries are currently not supported."
    from_idx = from_idxs[0]
    tokens_before_from = parsed_tokens[:from_idx]

    # split query into part between FROM and before GROUP BY (contain table names)
    # extract group by attribute
    group_by_idxs = [i for i, token in enumerate(parsed_tokens) if token.normalized == 'GROUP BY']
    assert len(group_by_idxs) == 0 or len(group_by_idxs) == 1, "Nested queries are currently not supported."
    group_by_attributes = None
    if len(group_by_idxs) == 1:  # there is a group by clause
        tokens_from_from = parsed_tokens[from_idx:group_by_idxs[0]]
        order_by_idxs = [i for i, token in enumerate(parsed_tokens) if token.normalized == 'ORDER BY']
        if len(order_by_idxs) > 0:
            group_by_end = order_by_idxs[0]
            tokens_group_by = parsed_tokens[group_by_idxs[0]:group_by_end]
        else:
            tokens_group_by = parsed_tokens[group_by_idxs[0]:]
        # Do not enforce single because there could be order by statement. Will be ignored.
        group_by_attributes = _extract_identifiers(tokens_group_by, enforce_single=False)
    else:
        tokens_from_from = parsed_tokens[from_idx:]

    # Get identifier to obtain relevant tables
    identifiers = _extract_identifiers(tokens_from_from)
    identifier_token_length = \
        [len(token.tokens) for token in identifiers if isinstance(token, sqlparse.sql.Identifier)][0]
    if identifier_token_length == 3:
        # (database, table)
        tables = [(token[0].value, token[2].value) for token in identifiers if
                  isinstance(token, sqlparse.sql.Identifier)]
    else:
        tables = [(token[0].value, token[0].value) for token in identifiers if
                  isinstance(token, sqlparse.sql.Identifier)]
    for database_name, table_name in tables:
        query_table=QueryTable(database_name,table_name)
        # query_table.columns=[c.name.upper() for c in catalog.listColumns(table_name,database_name)]
        query_table.columns=get_columns(table_name)
        query_table.add_sampling_method(sampling_methods[0])
        sampling_methods.pop(0)
        schema.add_table(query_table)
        query.add_table(query_table)

    # If there is a on clause, get the join condition
    on_idx = [idx for idx, token in enumerate(tokens_from_from) if token.normalized == 'ON']
    if len(on_idx) > 0:
        assert len(on_idx) == 1, "Nested queries are currently not supported."
        token_from_on = tokens_from_from[on_idx[0]:]
        on_statements = [token for token in token_from_on if isinstance(token, sqlparse.sql.Comparison)]
        assert len(on_statements) == 1, "On clause must be with join condition"
        on_statements = on_statements[0]
        left = on_statements.left
        right = on_statements.right
        assert isinstance(left, sqlparse.sql.Identifier), "Invalid where condition"
        assert isinstance(left, sqlparse.sql.Identifier), "Invalid where condition"

        comparison_tokens = [token for token in on_statements.tokens if token.ttype == Token.Operator.Comparison]
        assert len(comparison_tokens) == 1, "Invalid comparison"
        operator_idx = on_statements.tokens.index(comparison_tokens[0])
        assert on_statements.tokens[operator_idx].value == '=', "Invalid join condition"

        if len(left.tokens) == 1:  # when then join attribute without table name
            left_part=_extract_attribute(left,schema)
            # Join relationship
            assert len(right.tokens) == 1, "Invalid Identifier"
            right_part=_extract_attribute(right,schema)
            query.add_join_condition(left_part, right_part)
            
        else: # when the join attribute with the table name 
            # Replace alias by full table names
            left_part = _extract_attribute(left, schema)
            right = on_statements.right
            # Join relationship
            assert right.tokens[1].value == '.', "Invalid Identifier"
            right_part = _extract_attribute(right,schema)
            query.add_join_condition(left_part, right_part)
            
    # If there is a group by clause, parse it
    if group_by_attributes is not None:
        for group_by_token in _extract_identifiers(group_by_attributes):
            # attribute = group_by_token.value
            # table = _find_matching_table(attribute, schema)
            
            attribute=_extract_attribute(group_by_token,schema)
            query.add_group_by(attribute)

    # Obtain projection/aggregation attributes
    count_statements = [token for token in tokens_before_from if
                        token.normalized == 'COUNT(*)' or token.normalized == 'count(*)']
    assert len(count_statements) <= 1, "Several count statements are currently not supported."
    if len(count_statements) == 1:
        query.query_type = 0
    else:
        query.query_type = 1
        identifiers = _extract_identifiers(tokens_before_from)
        if isinstance(identifiers,sqlparse.sql.IdentifierList):
            # select group by attributes and aggregation attribute
            handle_aggregation(query, schema, identifiers)
        else:
            # select only aggregation attribute
            handle_aggregation(query, schema, tokens_before_from)

    # Obtain where statements
    where_statements = [token for token in tokens_from_from if isinstance(token, sqlparse.sql.Where)]
    assert len(where_statements) <= 1
    if len(where_statements) == 0:
        return query

    where_statements = where_statements[0]
    assert len(
        [token for token in where_statements if token.normalized == 'OR']) == 0, "OR statements currently unsupported."

    # normal comparisons condition
    comparisons = [token for token in where_statements if isinstance(token, sqlparse.sql.Comparison)]
    for comparison in comparisons:
        left = comparison.left
        assert isinstance(left, sqlparse.sql.Identifier), "Invalid where condition"
        comparison_tokens = [token for token in comparison.tokens if token.ttype == Token.Operator.Comparison]
        assert len(comparison_tokens) == 1, "Invalid comparison"
        operator_idx = comparison.tokens.index(comparison_tokens[0])

        if len(left.tokens) == 1:
            left_table_name, left_attribute = _fully_qualified_attribute_name(left, schema,
                                                                              return_split=True)
            left_part = left_table_name + '.' + left_attribute
            right = comparison.right
            # Join relationship
            if isinstance(right, sqlparse.sql.Identifier):
                assert len(right.tokens) == 1, "Invalid Identifier"

                right_attribute = right.tokens[0].value
                right_table_name=right_attribute
                # right_table_name = _find_matching_table(right_attribute, schema)
                right_part = right_table_name + '.' + right_attribute

                assert comparison.tokens[operator_idx].value == '=', "Invalid join condition"
                assert left_part + ' = ' + right_part in schema.relationship_dictionary.keys() or \
                       right_part + ' = ' + left_part in schema.relationship_dictionary.keys(), "Relationship unknown"
                if left_part + ' = ' + right_part in schema.relationship_dictionary.keys():
                    query.add_join_condition(left_part + ' = ' + right_part)
                elif right_part + ' = ' + left_part in schema.relationship_dictionary.keys():
                    query.add_join_condition(right_part + ' = ' + left_part)

            # Where condition
            else:
                where_condition = left_attribute + "".join(
                    [token.value.strip() for token in comparison.tokens[operator_idx:]])
                query.add_where_condition(left_table_name,left_attribute, where_condition)

        else:
            # Replace alias by full table names
            left_part = _fully_qualified_attribute_name(left, schema)

            right = comparison.right
            # Join relationship
            if isinstance(right, sqlparse.sql.Identifier):
                assert right.tokens[1].value == '.', "Invalid Identifier"
                right_part = right.tokens[0].value + '.' + right.tokens[2].value
                assert comparison.tokens[operator_idx].value == '=', "Invalid join condition"
                assert left_part + ' = ' + right_part in schema.relationship_dictionary.keys() or \
                       right_part + ' = ' + left_part in schema.relationship_dictionary.keys(), "Relationship unknown"
                if left_part + ' = ' + right_part in schema.relationship_dictionary.keys():
                    query.add_join_condition(left_part + ' = ' + right_part)
                elif right_part + ' = ' + left_part in schema.relationship_dictionary.keys():
                    query.add_join_condition(right_part + ' = ' + left_part)

            # Where condition
            else:
                query.add_where_condition(left.tokens[0].value,left.tokens[2].value,
                                          left.tokens[2].value + comparison.tokens[operator_idx].value + right.value)

    return query

def get_columns(table_name):
    table_name=table_name.lower()
    if table_name=='customer':
        return ['C_CUSTKEY', 'C_NAME', 'C_ADDRESS', 'C_NATIONKEY', 'C_PHONE', 'C_ACCTBAL', 'C_MKTSEGMENT', 'C_COMMENT']
    if table_name=='supplier':
        return ['S_SUPPKEY', 'S_NAME', 'S_ADDRESS', 'S_NATIONKEY', 'S_PHONE', 'S_ACCTBAL', 'S_COMMENT']
    if table_name=='store_sales':
        return ['SS_SOLD_DATE_SK', 'SS_SOLD_TIME_SK', 'SS_ITEM_SK', 'SS_CUSTOMER_SK', 'SS_CDEMO_SK', 'SS_HDEMO_SK', 'SS_ADDR_SK', 'SS_STORE_SK', 'SS_PROMO_SK', 'SS_TICKET_NUMBER', 'SS_QUANTITY', 'SS_WHOLESALE_COST', 'SS_LIST_PRICE', 'SS_SALES_PRICE', 'SS_EXT_DISCOUNT_AMT', 'SS_EXT_SALES_PRICE', 'SS_EXT_WHOLESALE_COST', 'SS_EXT_LIST_PRICE', 'SS_EXT_TAX', 'SS_COUPON_AMT', 'SS_NET_PAID', 'SS_NET_PAID_INC_TAX', 'SS_NET_PROFIT']
    if table_name=='web_sales':
        return ['WS_SOLD_DATE_SK', 'WS_SOLD_TIME_SK', 'WS_SHIP_DATE_SK', 'WS_ITEM_SK', 'WS_BILL_CUSTOMER_SK', 'WS_BILL_CDEMO_SK', 'WS_BILL_HDEMO_SK', 'WS_BILL_ADDR_SK', 'WS_SHIP_CUSTOMER_SK', 'WS_SHIP_CDEMO_SK', 'WS_SHIP_HDEMO_SK', 'WS_SHIP_ADDR_SK', 'WS_WEB_PAGE_SK', 'WS_WEB_SITE_SK', 'WS_SHIP_MODE_SK', 'WS_WAREHOUSE_SK', 'WS_PROMO_SK', 'WS_ORDER_NUMBER', 'WS_QUANTITY', 'WS_WHOLESALE_COST', 'WS_LIST_PRICE', 'WS_SALES_PRICE', 'WS_EXT_DISCOUNT_AMT', 'WS_EXT_SALES_PRICE', 'WS_EXT_WHOLESALE_COST', 'WS_EXT_LIST_PRICE', 'WS_EXT_TAX', 'WS_COUPON_AMT', 'WS_EXT_SHIP_COST', 'WS_NET_PAID', 'WS_NET_PAID_INC_TAX', 'WS_NET_PAID_INC_SHIP', 'WS_NET_PAID_INC_SHIP_TAX', 'WS_NET_PROFIT']
    if table_name=='movies':
        return ['M_MOVIEID', 'TITLE', 'GENRES']
    if table_name=='ratings':
        return ['USERID', 'R_MOVIEID', 'RATING', 'TSTAMP']
    if table_name=='genome_scores':
        return ['G_MOVIEID', 'TAGID', 'RELEVANCE']
    if table_name=='sdr_flow_with_outlier':
        return ['BATCHNO', 'STARTTIME', 'STR1', 'INT1', 'INT2', 'INT3', 'APN', 'PROT_CATEGORY', 'INT4', 'L4_UL_THROUGHPUT', 'L4_DW_THROUGHPUT', 'L4_UL_PACKETS', 'L4_DW_PACKETS', 'INT5', 'INT6', 'INT7', 'INT8', 'INT9', 'INT10', 'STR2', 'INT11', 'INT12']
    if table_name=='dim_sub_prot':
        return ['ID', 'SUB_PROT_ID', 'SUB_PROT', 'PROTOCOL_ID', 'PROTOCOL']
    return []